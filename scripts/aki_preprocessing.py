import argparse
import logging
import os
import pickle
import re
from collections import Counter, OrderedDict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--missing_path", type=str)
    parser.add_argument("--metadata_path", type=str)
    parser.add_argument("--feature_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def get_mutual_info_matrix(df):
    df = df.copy()
    df.replace("N/A", np.nan, inplace=True)
    remove_targets = ["Unique ID"]
    # Only Binary or Continuous
    discrete_cols = []
    continuous_cols = []

    df = df.drop(remove_targets, axis=1)
    for col in df.columns:
        if col in remove_targets:
            continue
        if df[col].nunique() <= 3:  # Binary
            uniques = [i for i in df[col].unique() if not pd.isna(i)]
            cat_type = CategoricalDtype(categories=uniques, ordered=True)
            df[col] = df[col].astype(cat_type)
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)

    with Pool(32, initializer=init_pool, initargs=(df, discrete_cols)) as p:
        res = p.map(calc_mutual_info, df.columns)

    res = pd.DataFrame(dict(res))
    res = res.reindex(res.columns)

    return res


def get_choices(df):
    df = df.replace("N/A", np.nan)
    col_dict = {}
    for col in df.columns:
        uniques = df[col].unique()
        float_keys = []
        string_keys = []
        floats, strings = [], []
        if len(uniques) <= 3:
            string_keys = list(uniques)
            strings = [str(i) for i in uniques]
        else:
            float_keys = list(set(uniques))
            floats = [float(i) for i in uniques if not pd.isna(i)]

        counts = dict(Counter(df[col][df[col].isin(string_keys)]))

        vocab, weights = [str(i) for i in counts.keys()], counts.values()
        weights = [i / sum(weights) for i in weights]

        float_ratio = len(floats) / (len(floats) + len(strings))

        res_dict = {
            "VOCAB": vocab,
            "ALL_VOCAB": set([str(i) for i in string_keys + float_keys]),
            "WEIGHTS": weights,
            "FLOAT_RATIO": float_ratio,
        }

        if len(floats) == 0:
            res_dict["GM"] = {
                "means": [0, 0, 0],
                "stds": [0, 0, 0],
                "weights": [0, 0, 0],
            }
        else:
            gm = GaussianMixture(n_components=3, random_state=42)
            all_floats = df[col][df[col].isin(float_keys)].map(float)
            gm.fit(np.array(all_floats).reshape(-1, 1))
            res_dict["GM"] = {
                "means": gm.means_.reshape(-1).tolist(),
                "stds": np.sqrt(gm.covariances_.reshape(-1)).tolist(),
                "weights": gm.weights_.tolist(),
            }

        col_dict[col] = res_dict

    return pd.DataFrame.from_dict(col_dict, orient="index")


def calc_mutual_info(target):
    df = df_global.copy()
    target_col = df[target]
    res = {}
    if isinstance(target_col.dtype, CategoricalDtype):
        target_col = target_col.cat.codes
        target_col.replace(-1, np.nan, inplace=True)

    train_target_notna = df[target_col.notna()]
    target_col = target_col[target_col.notna()]
    if len(target_col) < 3:
        return target, res
    is_target_discrete = target in discrete_cols_global

    for feature in df.columns:
        if target == feature:
            continue
        feature_col = train_target_notna[feature]
        if isinstance(feature_col.dtype, CategoricalDtype):
            feature_col = feature_col.cat.codes
        feature_col = feature_col.fillna(-1)

        is_feature_discrete = feature in discrete_cols_global

        if is_target_discrete:
            mutual_info = mutual_info_classif(
                np.expand_dims(feature_col.values, 1),
                target_col,
                discrete_features=is_feature_discrete,
            )[0]
        else:
            mutual_info = mutual_info_regression(
                np.expand_dims(feature_col.values, 1),
                target_col,
                discrete_features=is_feature_discrete,
            )[0]
        res[feature] = mutual_info
    return target, res


def init_pool(df, discrete_cols):
    global df_global
    global discrete_cols_global
    df_global = df
    discrete_cols_global = discrete_cols


def main():
    args = parse_args()

    # Log args
    logger.info(f"metadata_path: {args.metadata_path}")
    logger.info(f"data_path: {args.data_path}")
    logger.info(f"missing_path: {args.missing_path}")
    logger.info(f"output_path: {args.output_path}")
    logger.info(f"feature_path: {args.feature_path}")

    logger.info("Reading files...")
    data = pd.read_csv(args.data_path)
    missing_data = pd.read_csv(args.missing_path)
    metadata = pd.read_excel(args.metadata_path)
    feature_names = pd.read_csv(args.feature_path)

    # Part 1. Preprocess Metadata
    cols = metadata["컬럼명"]

    drop_list = []
    for col in cols:
        if re.match(r"d[1-8]_[1-3]_", col):
            if not col.startswith("d1_1"):
                drop_list.append(col)
        if re.match(r"d[1-8]_", col):
            if not col.startswith("d1"):
                drop_list.append(col)

    cols = set(cols) - set(drop_list)

    readables = metadata[metadata["컬럼명"].isin(cols)].reset_index(drop=True)

    readables["readable_name"] = feature_names["Human-readable Name"]

    # Part 2. Preprocess Data
    logger.info("Preprocessing data...")

    data = data.mask(missing_data == 0, "N/A")

    data = data[(data["icu"] == 1) & (data["ex_all"] == 0)].reset_index(drop=True)

    col_mapper = OrderedDict()
    for _, row in readables.iloc[4:10].iterrows():
        col_mapper[row["컬럼명"]] = (row["readable_name"], "[Baseline Characteristics]")
        if row["컬럼명"] == "sex":
            data[row["컬럼명"]].replace({1: "Male", 0: "Female"}, inplace=True)
        elif row["컬럼명"] == "icu":
            data[row["컬럼명"]].replace({1: "Yes", 0: "No"}, inplace=True)

    for _, row in readables.iloc[10:30].iterrows():
        col_mapper[row["컬럼명"]] = (row["readable_name"], "[Underlying Disease]")
        if row["컬럼명"] != "cci":
            data[row["컬럼명"]].replace({1: "Yes", 0: "No"}, inplace=True)

    for _, row in readables.iloc[31:46].iterrows():
        col_mapper[row["컬럼명"]] = (
            row["readable_name"],
            "[Prescription History within 6 Months Before Admission]",
        )
        data[row["컬럼명"]].replace({1: "Yes", 0: "No"}, inplace=True)

    for d in range(1, 8):
        for t in range(1, 4):
            for _, row in readables.iloc[67:98].iterrows():
                col_mapper[f"d{d}_{t}_" + row["컬럼명"][5:]] = (
                    row["readable_name"],
                    f"[Day {d} {(t-1)*8:02d}:00 - {t*8:02d}:00]",
                )

            # AKI, Critical AKI
            for _, row in readables.iloc[99:101].iterrows():
                col_mapper[f"d{d}_{t}_" + row["컬럼명"][5:]] = (
                    row["readable_name"],
                    f"[Day {d} {(t-1)*8:02d}:00 - {t*8:02d}:00]",
                )
                data[f"d{d}_{t}_" + row["컬럼명"][5:]].replace(
                    {1: "Yes", 0: "No"}, inplace=True
                )

            if t == 3:
                for _, row in readables.iloc[46:67].iterrows():
                    col_mapper[f"d{d}_" + row["컬럼명"][3:]] = (
                        row["readable_name"],
                        f"[Day {d}]",
                    )
                    if "surgery_time" not in row["컬럼명"]:
                        data[f"d{d}_" + row["컬럼명"][3:]].replace(
                            {1: "Yes", 0: "No"}, inplace=True
                        )

                # Dialysis
                col_mapper[f"d{d}_" + "dialysis"] = ("Dialysis", f"[Day {d}]")
                data[f"d{d}_" + "dialysis"].replace({1: "Yes", 0: "No"}, inplace=True)

    # Final outcomes
    for _, row in readables.iloc[102:111].iterrows():
        col_mapper[row["컬럼명"]] = (row["readable_name"], "[Final Outcomes]")
        if "cr_min" not in row["컬럼명"]:
            data[row["컬럼명"]].replace({1: "Yes", 0: "No"}, inplace=True)

    data.rename(columns={"Study_ID": "Unique ID"}, inplace=True)
    data = data[["Unique ID"] + list(col_mapper.keys())]

    mi_matrix = get_mutual_info_matrix(data)
    choices_df = get_choices(data)

    data = data.sample(n=1000, random_state=42).reset_index()

    os.makedirs(args.output_path, exist_ok=True)
    data.to_csv(f"{args.output_path}/valid.csv", index=False)
    choices_df.to_pickle(f"{args.output_path}/choices.pkl")
    mi_matrix.to_pickle(f"{args.output_path}/mi_matrix.pkl")
    with open(f"{args.output_path}/col_mapper.pkl", "wb") as f:
        pickle.dump(col_mapper, f)
    logger.info("Done!")

    return


if __name__ == "__main__":
    main()
