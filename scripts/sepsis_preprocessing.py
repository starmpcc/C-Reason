import argparse
import datetime
import logging
import os
import pickle
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

# Units to use:
active_units = [
    "cm",
    "kg",
    "kg/m2",
    "mmHg",
    "/min",
    "℃",
    "mmol/L",
    "∙103/uL",
    "%",
    "/uL",
    "g/dL",
    "mg/dL",
    "U/L",
    "INR",
    "mg/L",
    "ng/mL",
    "mcg/mL",
    "pg/mL",
    "103/uL",
    "mL",
    "Days",
    "Minutes",
]

# Units to replace
unit_to_replace = {
    "kg/m2": "kg/m^2",
    "∙103/uL": "10^3/uL",
    "103/uL": "10^3/uL",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--kor_to_eng_dict_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def parse_domain_section_headers(metadata):
    domain_list = metadata["Domain"].tolist()
    section_list = metadata["Section"].tolist()

    domain_section_kor_to_eng_dict = {
        "환자 및 기관 정보": "Patient and Institution Information",
        "선정기준 및 기초자료": "Selection Criteria and Basic Data",
        "스크리닝기준": "Screening Criteria",
        "연구대상자    선정기준": "Selection Criteria for Study Participants",
        "연구대상자의 기초자료": "Basic Data of Study Participants",
        "bundles 시간 관련 변수": "Bundles Time-related Variables",
        "패혈증 묶음 치료 관련 변수": "Variables Related to Sepsis Bundle Treatment",
        "중환자실 관련 시간 변수": "ICU-related Time Variables",
        "자료 차수 구분자": "Data Batch Identifier",
    }

    domain_section_list = []
    current_domain = None
    current_section = None
    for domain, section in zip(domain_list, section_list):
        if pd.notna(domain):
            if domain in domain_section_kor_to_eng_dict:
                domain = domain_section_kor_to_eng_dict[domain]
            current_domain = domain.strip().replace("\n", " ")
        if pd.notna(section):
            if section in domain_section_kor_to_eng_dict:
                section = domain_section_kor_to_eng_dict[section]
            current_section = section.strip().replace("\n", " ")

        domain_section_list.append(f"[{current_domain}/{current_section}]")
    return domain_section_list


def parse_range_to_categorical(range_str, kor_to_eng_dict):
    if "\n" not in str(range_str):
        if groups := re.findall(r"\[(\d)\] (.*)", str(range_str)):
            number, text = groups[0]
            if number.isdigit():
                number = int(number)
            text = text.replace("=", "").replace(".", "").strip()
            # 한 줄일 경우 Binary Yes 에 해당
            return {number: "Yes", 0: "No"}

        return None

    categorical_dict = {}
    ranges = range_str.split("\n")

    for idx, range in enumerate(ranges):
        if range.startswith("["):
            texts = re.findall(r"\[([A-Z0-9]+)\](.*)", range)
        elif re.match(r"^\d", range):
            texts = re.findall(r"(\d+)\s?=\s?(.*)", range)

        if len(texts) == 0 or len(texts[0]) != 2:
            if idx == 0:
                continue
            return None

        key, val = texts[0]
        if key.isdigit():
            key = int(key)
        val = val.strip()
        if val in kor_to_eng_dict:
            val = kor_to_eng_dict[val]
        categorical_dict[key] = val

    if not categorical_dict:
        return None
    return categorical_dict


def row_with_unit(name, unit):
    name = name.strip()
    if unit not in active_units:
        return name

    if unit in unit_to_replace:
        unit = unit_to_replace[unit]
    return f"{name} ({unit})"


def int_to_str_keys(categorical_dict):
    if categorical_dict is None:
        return None
    for k in list(categorical_dict.keys()):
        if isinstance(k, int):
            categorical_dict[str(k)] = categorical_dict[k]
    return categorical_dict


def parse_mv_pat(row):
    # MVNOPAT: only 3,4
    # MVPAT: only 1,2
    if row["MVPAT"] in ["3", "4"]:
        return "NA", row["MVPAT"]
    elif row["MVPAT"] in ["1", "2"]:
        return row["MVPAT"], "NA"
    elif row["MVNOPAT"] in ["3", "4"]:
        return "NA", row["MVNOPAT"]
    elif row["MVNOPAT"] in ["1", "2"]:
        return row["MVNOPAT"], "NA"
    else:
        return row["MVPAT"], row["MVNOPAT"]


def map_unqiues(col, mapping_dict):
    def map_value(x, col_name, mapping_dict):
        if isinstance(x, int):
            x = str(x)
        if x == "":
            return None
        elif x == "NA":
            return "NA"
        elif x in mapping_dict:
            return mapping_dict[x]
        elif x.endswith(","):
            x = x[:-1]
            return map_value(x, col_name, mapping_dict)
        elif "," in x:
            # Multiple case
            x_list = x.split(",")
            try:
                res = [mapping_dict[i.strip()] for i in sorted(x_list)]
                return ", ".join(res)
            except:
                raise Exception(f"Error on {x}, {col_name}, {mapping_dict}")
        elif x == "No":
            return "No"
        else:
            raise Exception(f"Error on {x}, {col_name}, {mapping_dict}")

    res_dict = {}
    for unique in col.unique():
        res_dict[unique] = map_value(unique, col.name, mapping_dict)

    return res_dict


def nan_to_none(x, col, df):
    if col in [
        "liver_disease",
        "solid_tumor",
        "chronic_kidney_ds",
    ]:
        return x.replace("None", "no").fillna("no")
    elif col in [
        "remvinfectintravas",
        "inspercutdrain",
        "othernonsurg",
    ]:
        x.mask(x.isna() & (df["firstsourcenonsurg"] == "yes"), "no")
    elif col.startswith("bacmttyp"):
        return x.mask(x.isna() & (df["bacmtyn"] == "yes"), "no")
    elif col in ["gramposyn", "gramnegyn", "atybacyn", "bacmtyn"]:
        return x.mask(x.isna() & (df["bactathogenmet"] == "yes"), "no")
    else:
        return x


def get_choices(df):
    col_dict = {}
    for col in df.columns:
        uniques = df[col].unique()
        float_keys = []
        string_keys = []
        if len(uniques) <= 1:
            continue
        floats, strings = [], []
        for i in uniques:
            try:
                floats.append(float(i))
                float_keys.append(i)
            except:
                if i:
                    strings.append(i)
                    string_keys.append(i)
        if len(set(floats)) <= 20:
            strings += [str(i) for i in floats]
            floats = []
            string_keys += float_keys
            float_keys = []

        if len(strings) > 20:
            if re.search(r"date|time|dat_ou", col) or col == "tzdt":
                logger.info(f"Skipping Date/Time col: {col}")
                continue
            if col.endswith("desc") or col in [
                "Unique ID",
                "sepsisinkcd",
                "othernonsurgprocedure",
                "operationname",
                "gramposot",
                "gramnegot",
                "bacposspecot",
                "fugusot",
            ]:
                logger.info(f"Skipping Text col: {col}")
                continue
            else:
                logger.info(f"Multi-Choice col: {col}")

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


def get_mutual_info_matrix(df):
    df = df.copy()
    df["tzdt"] = pd.to_datetime(df["tzdt"], format="mixed")
    df.replace("", np.nan, inplace=True)
    remove_targets = ["Unique ID", "tzdt"]
    discrete_cols = []
    continuous_cols = []

    continuouts_categorical_dict = {
        "age_charlson": ["<50", "50-59", "60-69", "70-79", "≥80"],
        "hoslengthbficu": ["<14", "≥ 14 < 28", "≥ 28"],
        "hspt_bd": ["<500", "501~1000", "1001~1500", ">1500"],
        "hos_rrs_grade": ["Not applicable", "Grade 1", "Grade 2", "Grade 3"],
        "hos_rrs_time": [
            "Not applicable",
            "<8 hours/day",
            "8~16 hours/day",
            "16~24 hours/day",
            "24 hours/day",
        ],
        "ecog_initial": ["0", "1", "2", "3", "4"],
        "ecog_final": ["0", "1", "2", "3", "4", "5"],
        "cfscore": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "clinfrailtyscore": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    }

    for col in df.columns:
        if col in remove_targets:
            continue
        if df[col].dtype in ["object", "datetime64[ns]"]:
            if df[col].nunique() <= 30:
                uniques = [i for i in df[col].unique() if not pd.isna(i)]
                cat_type = CategoricalDtype(categories=uniques, ordered=True)
                df[col] = df[col].astype(cat_type)

                if col in [
                    "age_charlson",
                    "hoslengthbficu",
                    "hspt_bd",
                    "hos_rrs_grade",
                    "hos_rrs_time",
                    "ecog_initial",
                    "ecog_final",
                    "cfscore",
                    "clinfrailtyscore",
                ]:
                    df[col] = df[col].cat.reorder_categories(
                        continuouts_categorical_dict[col], ordered=True
                    )

                    continuous_cols.append(col)
                else:
                    discrete_cols.append(col)

            else:
                if re.search(r"date|time|dat_ou", col):
                    df[col] = pd.to_datetime(df[col], yearfirst=True, format="mixed")
                    df[col] = (
                        df[col] - df["tzdt"]
                    ).dt.total_seconds() // 60  # to minutes
                    continuous_cols.append(col)
                else:
                    logger.info(f"MI Skipping col {col}, Nunique: {df[col].nunique()}")
                    remove_targets.append(col)
        else:
            continuous_cols.append(col)

    df = df.drop(remove_targets, axis=1)

    with Pool(32, initializer=init_pool, initargs=(df, discrete_cols)) as p:
        res = p.map(calc_mutual_info, df.columns)

    res = pd.DataFrame(dict(res))
    res = res.reindex(res.columns)

    return res


def main():
    args = parse_args()

    # Log args
    logger.info(f"metadata_path: {args.metadata_path}")
    logger.info(f"data_path: {args.data_path}")
    logger.info(f"output_path: {args.output_path}")
    logger.info(f"kor_to_eng_dict_path: {args.kor_to_eng_dict_path}")

    # Read files
    logger.info("Reading files...")
    metadata = pd.read_excel(args.metadata_path)
    data = pd.read_excel(args.data_path, keep_default_na=False)
    with open(args.kor_to_eng_dict_path, "rb") as f:
        kor_to_eng_dict = pickle.load(f)

    # Part 1. Preprocess metadata
    logger.info("Preprocessing metadata...")
    metadata["DOMAIN_SECTION_HEADER"] = parse_domain_section_headers(metadata)

    # Some of category info exists on decision column
    metadata.loc[688:, "Range"] = metadata.loc[688:, "Decision"]
    metadata["categorical_dict"] = metadata["Range"].apply(
        parse_range_to_categorical, kor_to_eng_dict=kor_to_eng_dict
    )

    metadata["row_name_with_unit"] = metadata.apply(
        lambda x: row_with_unit(x["Row Name"], x["Unit"]), axis=1
    )

    metadata["categorical_dict"] = metadata["categorical_dict"].map(int_to_str_keys)

    metadata["eCRF Variable name "] = [
        i.lower() for i in metadata["eCRF Variable name "]
    ]

    # Part 2. Preprocess data
    logger.info("Preprocessing data...")
    data["MVPAT"], data["MVNOPAT"] = zip(*data.apply(parse_mv_pat, axis=1))

    data.columns = [i.lower() for i in data.columns]
    for col in data.columns:
        if data[col].dtype == "object":
            str_uniques = [i for i in data[col].unique() if isinstance(i, str)]
            lower_uniques = pd.Series(str_uniques).str.lower()
            dups = lower_uniques[lower_uniques.duplicated(keep=False)]
            if len(dups) > 0:
                logger.info(f"Lowering duplicates: {col}, {set(dups.tolist())}")
                data[col] = data[col].map(
                    lambda x: (
                        x.lower() if isinstance(x, str) and x.lower() in dups else x
                    )
                )

    for col in data.columns:
        remaining_keys = set(data[col].unique()) - set(["", "NA"])
        if len(remaining_keys) == 1:
            remaining_key = remaining_keys.pop()
            if "_icud" in col and "na_" not in col:
                pivot = "na_icud" + col.split("_icud")[1]

                def _yes_na_mapper(x, col, pivot):
                    if x[col] == remaining_key:
                        return x[col]
                    elif x[pivot] == "NA":
                        return ""
                    else:
                        return "No"

                data[col] = data.apply(_yes_na_mapper, args=(col, pivot), axis=1)
                logger.info(f"Fill NA to No: {col}")

    # Part 3. Merge metadata and data
    logger.info("Merging metadata and data...")
    # Some typo in columns are also handeled
    manual_mapping = dict(
        zip(
            sorted(list(set(metadata["eCRF Variable name "]) - set(data.columns))),
            sorted(list(set(data.columns) - set(metadata["eCRF Variable name "]))),
        )
    )

    metadata["eCRF Variable name "] = metadata["eCRF Variable name "].map(
        lambda x: manual_mapping.get(x, x)
    )

    data = data[metadata["eCRF Variable name "]]

    # Remove 'datano' column
    metadata = metadata[:-1]
    data = data.drop(columns="datano")

    mapping_dict = {
        data.columns[i]: map_unqiues(
            data[data.columns[i]], metadata.iloc[i]["categorical_dict"]
        )
        for i in range(len(metadata))
        if metadata.iloc[i]["categorical_dict"] is not None
    }
    data.replace(mapping_dict, inplace=True)
    data.replace(kor_to_eng_dict, inplace=True)

    # Nan to None
    data.replace(
        {
            "NA": np.nan,
            "Yes": "yes",
            "No": "no",
        },
        inplace=True,
    )
    for col in data.columns:
        data[col] = nan_to_none(data[col], col, data)

    # Remove "Derived/buldles 시간 관련 변수" columns
    datetime_cols = [i for i in data.columns if "datetime" in i or i == "tztmtime"]
    data.drop(columns=datetime_cols, inplace=True)
    metadata.drop(
        metadata[metadata["eCRF Variable name "].isin(datetime_cols)].index,
        inplace=True,
    )
    metadata.reset_index(drop=True, inplace=True)

    drop_cols = [i for i in data.columns if i.startswith("na_")]
    drop_cols += ["hosadmna", "hspt_cd"]
    for i in range(len(data.columns)):
        # Date + Time -> Datetime
        if (
            "date" in metadata["row_name_with_unit"][i].lower()
            and "time" in metadata["row_name_with_unit"][i + 1].lower()
        ):
            date = data[data.columns[i]]
            time = data[data.columns[i + 1]]
            new_value = [
                datetime.datetime.combine(d, t) if d != "" and t != "" else None
                for d, t in zip(date, time)
            ]
            data[data.columns[i]] = new_value
            drop_cols.append(data.columns[i + 1])
            metadata.loc[
                metadata["eCRF Variable name "] == data.columns[i], "row_name_with_unit"
            ] += "time"
            logger.info(
                f"Converted to datetime: {data.columns[i]}, {data.columns[i + 1]}"
            )

    data.drop(columns=drop_cols, inplace=True)
    metadata.drop(
        metadata[metadata["eCRF Variable name "].isin(drop_cols)].index, inplace=True
    )

    col_mapper = dict(
        [
            (
                i["eCRF Variable name "],
                (i["row_name_with_unit"], i["DOMAIN_SECTION_HEADER"]),
            )
            for _, i in metadata.iterrows()
            if i["eCRF Variable name "] not in ["subjectno", "center"]
        ]
    )

    # Part 7. Save to disk
    logger.info("Saving to disk...")
    data = data.rename(columns={"subjectno": "Unique ID"})

    target_cols = list(col_mapper.keys()) + ["Unique ID"]
    data = data[[i for i in data.columns if i in target_cols]]

    mi_matrix = get_mutual_info_matrix(data)
    # Gaussianmixture mp can cause mp error. Should do after mi
    choices_df = get_choices(data)

    train, valid = train_test_split(data, test_size=1000, random_state=42)

    os.makedirs(args.output_path, exist_ok=True)
    train.to_csv(f"{args.output_path}/train.csv", index=False)
    valid.to_csv(f"{args.output_path}/valid.csv", index=False)
    choices_df.to_pickle(f"{args.output_path}/choices.pkl")
    mi_matrix.to_pickle(f"{args.output_path}/mi_matrix.pkl")
    with open(f"{args.output_path}/col_mapper.pkl", "wb") as f:
        pickle.dump(col_mapper, f)
    logger.info("Done!")


if __name__ == "__main__":
    main()
