import argparse
import logging
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

coding_manual_mapping_dict = {
    "남자-1 여자-0": {0: "Female", 1: "Male"},
    "1=ischemic stroke  2=hemorrhagic stroke  3=TIA": {
        1: "Ischemic Stroke",
        2: "Hemorrhagic Stroke",
        3: "TIA",
    },
    "NA=999": {999: "NA"},
    "1=yes": {1: "Yes"},
    "1=hemorrhagic/2=ischemic/3=mixed/4=unknown": {
        1: "Hemorrhagic",
        2: "Ischemic",
        3: "Mixed",
        4: "Unknown",
    },
    "1=current smoker/2=quit>=5y /3=quit<5y/ 4=never": {
        1: "Current Smoker",
        2: "Quit >= 5y",
        3: "Quit < 5y",
        4: "Never",
    },
    "1=예": {1: "Yes"},
    "1=expired/2=hopeless discharge/ 3=DAMA/ 4=transfer to other department/5=discharge": {
        1: "Expired",
        2: "Hopeless Discharge",
        3: "DAMA",
        4: "Transfer to Other Department",
        5: "Discharge",
    },
    "1=directly related to stroke/ 2=indirectly related/3=unknown": {
        1: "Directly Related to Stroke",
        2: "Indirectly Related",
        3: "Unknown",
    },
    "1=to home/ 2=referred to": {1: "To Home", 2: "Referred to"},
    "1=예/ 0=아니오": {1: "Yes", 0: "No"},
    "1=예 0=아니오 9=NA": {1: "Yes", 0: "No", 9: "NA"},
    "1=본원. 2= 타병원": {1: "This Hospital", 2: "Other Hospital"},
    "ER-2, OPD-1  in-hospital-3": {1: "OPD", 2: "ER", 3: "in-hospital"},
    "1=LAA/2=SVO/3=CE/4=Other determined/5=undetermined 2 or more, 6=negative, 7=incomplete": {
        1: "LAA",
        2: "SVO",
        3: "CE",
        4: "Other determined",
        5: "undetermined",
        6: "negative",
        7: "incomplete",
    },
    "Rt=1, Lt=2 Both=3": {1: "Rt", 2: "Lt", 3: "Both"},
    "1=<50% stenosis, 2=>=50% stenosis, 3=occlusion": {
        1: "<50% stenosis",
        2: ">=50% stenosis",
        3: "occlusion",
    },
    "1=complete occlusion (TIMI  0)  2=partial occlusion (TIMI 1,2) 3=no occlusion (TIMI 3) 4=미확인 9=알수없음": {
        1: "complete occlusion (TIMI 0)",
        2: "partial occlusion (TIMI 1,2)",
        3: "no occlusion (TIMI 3)",
        4: "Unconfirmed",
        9: "Unknown",
    },
    "1=no recanalization (TIMI 0)  2=partial recanalization (TIMI 1,2)  3=complete recanalization (TIMI 3) 4=미확인 9=알수없음": {
        1: "no recanalization (TIMI 0)",
        2: "partial recanalization (TIMI 1,2)",
        3: "complete recanalization (TIMI 3)",
        4: "Unconfirmed",
        9: "Unknown",
    },
    "1=Stroke recurrence, 2=Stroke progression\n3=Sym HT,  4= Others, 5=unknown, 6=tia": {
        1: "Stroke recurrence",
        2: "Stroke progression",
        3: "Sym HT",
        4: "Others",
        5: "unknown",
        6: "tia",
    },
    "1=Angina, 2=AMI, 3= Congestive heart failure\n5=unknown ": {
        1: "Angina",
        2: "AMI",
        3: "Congestive heart failure",
        5: "unknown",
    },
    'NA="."': {".": "NA"},
}

coding_kor_to_eng_dict = {
    "무학/문맹": "Illiterate",
    "내원 1시간 이내 CPR 시행": "CPR performed within 1 hour of arrival",
    "내원 직전 외부에서 촬영한 뇌영상 있음": "Brain imaging done outside just before arrival",
    "병원 1시간 이내 증상소실(NIHSS 0)": "Symptoms disappear within 1 hour of hospital (NIHSS 0)",
    "기타": "Etc.",
    "이유없음": "No reason",
    "시행안함": "Not performed",
    "시간이 늦은 타당한사유가 있음": "Valid reason for being late",
    "시간 확인했으나 알수없음": "Time confirmed but unknown",
    "증상발생시간 및 최종정상확인 시각을 모르는 경우": "If the time of symptom onset and the last normal check time are unknown",
    "피검사에서 출혈성향이 있는경우 (혈소판 <10K or abnormal PTT or INR {PT) >1.5": "In case of bleeding tendency in blood tests (Platelets <10K or abnormal PTT or INR {PT) >1.5)",
    "조절되지 않은 고혈압": "Uncontrolled hypertension",
    "환자 또는 보호자의 거부": "Patient or guardian's refusal",
    "타병원에서 정맥내 혈전용해제 투여후 전원": "Transferred after intravenous thrombolytic therapy in another hospital",
    "미확인": "Unconfirmed",
    "알수없음": "Unknown",
    "동의": "Agree",
    "동의안함": "Do not agree",
    "본인": "Self",
    "가족 (텍스트": "Family (text)",
    "비싸서": "Because it's expensive",
    " 부작용": "Side effects",
    " 효과없음": "Ineffective",
    "기타 (텍스트)": "Others (text)",
    "무학": "Illiterate",
    "출혈의 위험성": "Risk of bleeding",
    "경미한 증상 또는 증상의 급속한 호전": "Mild symptoms or rapid improvement of symptoms",
    "다른 질환으로 인한 것일 가능성이 있는 경우": "If it may be due to another disease",
    "60분 이내 시행": "Performed within 60 minutes",
    "60분 이후 시행": "Performed after 60 minutes",
    "증상이 호전되다 악화되는 경우": "If symptoms improve and then worsen",
    "표준진료지침의 권고이상의 혈압으로 혈압강하치료가 우선된 경우": "If blood pressure lowering treatment is prioritized due to higher than recommended blood pressure in standard treatment guidelines",
    "호흡곤란이나 활력징후가 불안정하여 기도삽관이 우선 시행되었던 경우": "If airway intubation was prioritized due to breathing difficulties or unstable vital signs",
    "타병원시행하여 약종류 모름": "Performed at another hospital, type of medication unknown",
    "확인": "Confirmed",
    "아니오": "No",
    "예": "Yes",
    " 가족 (텍스트": "Family (text)",
    "0-13년이상": "0-13 years",
    "0-3년": "0-3 years",
    "4-6년": "4-6 years",
    "7-9년": "7-9 years",
    "10-12년": "10-12 years",
    "13년 이상": "13 years or more",
    "무응답": "No response",
    "상세점수 없음 or 점수 틀림": "No detailed score or wrong score",
    "부분점수만 들어있어 0으로 채움": "Some partial scores exist and others filled with 0",
    "모두 입력되어있음": "All exist",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--metadata_path", type=str)
    parser.add_argument("--nihss_path", type=str)
    parser.add_argument("--additional_path", type=str)
    parser.add_argument("--kor_to_eng_dict_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def parse_metadata_section(metadata):
    metadata_dict = {}
    start, end = 0, 0
    for idx, row in metadata.iterrows():
        # Section start
        if (
            all(pd.isna(row))
            or row[0]
            == "*\uf054 충북대병원 pre mrs: ~ 2019.9 까지 adm mrs 로 기록되어 있음.  2017년 자료까지 999로 처리함"
        ):
            end = idx
            if metadata.iloc[start, 0] in metadata_dict:
                continue
            metadata_dict[metadata.iloc[start, 0]] = metadata.iloc[start + 1 : end]
        elif pd.notna(row[0]) and all(pd.isna(row[1:])):
            start = idx
    end = idx + 1
    metadata_dict[metadata.iloc[start, 0]] = metadata.iloc[start + 1 : end]

    for k in metadata_dict.keys():
        metadata_dict[k].columns = metadata_dict[k].iloc[0]
        metadata_dict[k] = metadata_dict[k].iloc[1:].reset_index(drop=True)
        metadata_dict[k]["Section"] = k

    metadata = pd.concat(metadata_dict.values(), ignore_index=True)

    metadata = metadata[
        [
            "Section",
            "필드정의",
            "SNUBH field name",
            "field character",
            "coding",
            "시작시점",
            "Row Name",
        ]
    ]

    return metadata


wrong_code_mapper = {
    "im-pos": "im_pos",
    "NIH1d": "nih_1d",
    "f_img": "fimg",
    "f_img_sht": "sHT",
    "atx_dabig": "atx_dabi",
    "aorta_ct": "aorct",
    "trans_sub1": "trans_sub_1",
    "end_swelling": "end1_swelling",
}


def parse_coding_to_mapping_dict(x):
    out = None
    if x in coding_manual_mapping_dict:
        return coding_manual_mapping_dict[x]
    elif "," not in str(x):
        return None

    splitteds = x.split(",")

    for sep in ["=", ".", "-"]:
        if all([(sep in j) for j in splitteds]):
            try:
                out = {b: a for b, a in [j.split(sep) for j in splitteds]}
            except:
                pass
            else:
                break

    if out is None:
        return None

    for k in list(out.keys()):
        if out[k] in coding_kor_to_eng_dict:
            out[k] = coding_kor_to_eng_dict[out[k]]
        out[k] = out[k].strip()
        if isinstance(k, str) and k != k.strip():
            out[k.strip()] = out.pop(k)
            k = k.strip()
        if isinstance(k, str) and k.isdigit():
            out[int(k)] = out.pop(k)
    return out


def edu_mapper(edu1, edu2):
    if edu1 == "":
        return None
    if edu1 == 999 or edu2 == 999:
        return "NA"
    if edu1 in [0, 1]:
        return "illiterate/uneducated"
    elif edu1 == 2:
        return "uneducated"
    elif edu1 == 3:
        if edu2 == "":
            return "NA"
        if edu2 == 1:
            return "0-3 years"
        elif edu2 == 2:
            return "4-6 years"
        elif edu2 == 3:
            return "7-9 years"
        elif edu2 == 4:
            return "10-12 years"
        elif edu2 == 5:
            return "13 years or more"
        elif edu2 == 9:
            return "No response"
    raise ValueError(f"edu1: {edu1}, edu2: {edu2}")


nan_to_no_cols = [
    "l_multi",
    "l_n",
    "a_aca_s",
    "a_mca_s",
    "a_pca_s",
    "a_ba_s",
    "a_va_s",
    "a_exica_s",
    "a_inica_s",
    "a_cca_s",
    "a_aort_s",
    "a_multi",
    "a_n",
    "ivtpa",
    "iv_tpadose",
    "nih_1d",
    "occl_state",
    "recan_state",
    "tf_in",
    "ivh",
    "sah",
    "sdh",
    "tia",
    "im_pos",
    "ia_uro",
    "ia_reoper",
    "ia_tirof",
    "ia_drug_o",
    "ia_penumbra",
    "ia_solitare",
    "ia_merci",
    "no_end",
    "end",
    "end2",
    "end3",
    "con_loss_3m",
    "ava_no",
    "no_event3m",
    "ev1_3m",
    "ev2_3m",
    "ev3_3m",
    "con_loss_1y",
    "no_event1y",
    "ev1_1y",
    "ev2_1y",
]


def map_values(x, mapping_dict, col):
    if col in nan_to_no_cols:
        if pd.isna(x) or x in ["Yes", "yes", 1.0, True]:
            return "Yes"
        else:
            return "No"
    if pd.isna(x) or x == "":
        return None
    if isinstance(x, str) and x.isdigit() and int(x) in [999, 9999]:
        return "NA"
    elif isinstance(x, (int, float)) and x in [999, 9999]:
        return "NA"
    if mapping_dict is None:
        return x
    else:
        if x in mapping_dict:
            return mapping_dict[x]
        elif isinstance(x, str) and x.isdigit() and int(x) in mapping_dict:
            return mapping_dict[int(x)]
        # 여기부터 edge case들
        elif mapping_dict.get(1) in ["yes", "Yes"]:
            if x == 0:
                return "no"
            elif col == "ce_medium" and x == 2:
                return None
            elif x == 9:
                return "NA"
            else:
                raise ValueError(f"col: {col}, x: {x}, mapping_dict: {mapping_dict}")
        else:
            if len(mapping_dict) == 1 and "NA" in mapping_dict.values():
                return x
            if col == "ivtpa_n" and x == 11:
                return mapping_dict[10]
            elif col == "ivtpa60_n" and x == 5:
                return mapping_dict[4]
            elif col == "post_evt_state" and x == "미확인":
                return "Unconfirmed"
            elif x == 9:
                return "NA"
            else:
                return None


def get_choices(df, metadata):
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
            if col in metadata["SNUBH field name"].values and (
                metadata.loc[
                    metadata["SNUBH field name"] == col, "field character"
                ].values[0]
                == "날짜/시간"
            ):
                logger.info(f"Skipping Date/Time col: {col}")
                continue
            if (
                "_ow" in col
                or col.endswith("_nw")
                or col
                in [
                    "Unique ID",
                    "cc",
                    "image_nw",
                    "psce1_12_ow",
                    "in_txt",
                    "comment",
                    "drug_stop_w",
                    "ev1_odw_3m",
                    "ev1_hosw_3m",
                    "ev1_odw_1y",
                    "ev1_hosw_1y",
                    "trans_sub_1",
                    "dis_sub_1",
                ]
            ):
                logger.info(f"Skipping Text col: {col}")
                continue
            else:
                logger.info(f"Multi-Choice col: {col}")
            # raise NotImplementedError
            # logger.info(f"NotImplemented: {col}")
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

    if is_target_discrete:
        if target_col.value_counts().min() < 3:
            return target, res

    for feature in df.columns:
        if target == feature:
            continue
        feature_col = train_target_notna[feature]
        if isinstance(feature_col.dtype, CategoricalDtype):
            feature_col = feature_col.cat.codes
        feature_col = feature_col.fillna(-1)

        is_feature_discrete = feature in discrete_cols_global

        if is_feature_discrete:
            if feature_col.value_counts().min() < 3:
                continue

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
    df = df.replace(
        {
            "NA": np.nan,
            "Not measurable": np.nan,
            "Unknown": np.nan,
            "Not necessary": np.nan,
            "": np.nan,
        },
    )

    df["arrival"] = pd.to_datetime(df["arrival"], format="mixed")
    remove_targets = ["Unique ID"]
    discrete_cols = []
    continuous_cols = []

    for col in df.columns:
        if col in remove_targets:
            continue
        if df[col].notna().sum() == 0:
            remove_targets.append(col)
            continue
        if df[col].dtype in ["object", "datetime64[ns]"]:
            if df[col].nunique() <= 30:
                uniques = [i for i in df[col].unique() if not pd.isna(i)]
                cat_type = CategoricalDtype(categories=uniques, ordered=True)
                df[col] = df[col].astype(cat_type)
                discrete_cols.append(col)

            else:
                try:
                    df[col] = pd.to_datetime(df[col], format="mixed")
                    df[col] = (
                        df[col] - df["arrival"]
                    ).dt.total_seconds() // 60  # to minutes
                    continuous_cols.append(col)
                except:
                    logger.info(f"MI Skipping col {col}, Nunique: {df[col].nunique()}")
                    remove_targets.append(col)
        else:
            continuous_cols.append(col)

    df = df.drop(remove_targets, axis=1)

    init_pool(df, discrete_cols)
    res = []
    for col in df.columns:
        res.append(calc_mutual_info(col))

    res = pd.DataFrame(dict(res))
    res = res.reindex(res.columns)

    return res


def main():
    args = parse_args()

    # Log args
    logger.info(f"metadata_path: {args.metadata_path}")
    logger.info(f"data_path: {args.data_path}")
    logger.info(f"nihss_path: {args.nihss_path}")
    logger.info(f"kor_to_eng_dict_path: {args.kor_to_eng_dict_path}")
    logger.info(f"output_path: {args.output_path}")

    # Read files
    logger.info("Reading files...")
    metadata = pd.read_excel(args.metadata_path, header=None)
    data = pd.read_excel(args.data_path, keep_default_na=False)
    nihss = pd.read_excel(args.nihss_path, keep_default_na=False)
    additional = pd.read_excel(args.additional_path, keep_default_na=False)

    with open(args.kor_to_eng_dict_path, "rb") as f:
        kor_to_eng_dict = pickle.load(f)

    # Part 1. Preprocess Metadata
    logger.info("Preprocessing Metadata...")

    metadata = parse_metadata_section(metadata)
    metadata = metadata[metadata["Row Name"].notna()]

    # Handle errors in metadata
    metadata.loc[
        metadata["필드정의"] == "Risk Factors(dm 유 무 ", "SNUBH field name"
    ] = "hx_dm"

    metadata["SNUBH field name"] = metadata["SNUBH field name"].replace(
        wrong_code_mapper
    )

    # Parse coding to dict
    metadata["mapping_dict"] = metadata["coding"].map(parse_coding_to_mapping_dict)

    metadata["SNUBH field name"] = metadata["SNUBH field name"].str.lower()

    # Part 2. Preprocess data & Filter Cohort
    data = pd.merge(data, nihss, on="고유번호", how="left")
    data = pd.merge(
        data, additional, left_on="고유번호", right_on="uni_num", how="left"
    )

    data.columns = [i.lower() for i in data.columns]

    # Remove error columns
    data = data.drop(columns=["htx_doac_d", "htx_doac_dw"])

    # Map "edu1" and "edu2" to "edu"
    data["edu1"] = data.apply(lambda x: edu_mapper(x["edu1"], x["edu2"]), axis=1)
    data.drop(columns=["edu2"])
    data.rename(columns={"edu1": "edu"}, inplace=True)

    # Map Angiography Columns
    angiography_keys = [i for i in metadata["SNUBH field name"] if i.startswith("a_")]
    angiography_keys = [i[:-2] for i in angiography_keys if i.endswith("_c")]

    for k in angiography_keys:
        data[k], data[k + "_c"], data[k + "_s"] = zip(
            *data.apply(
                lambda x: (
                    (0, None, None)
                    if x[k + "_s"] == 0
                    else (x[k], x[k + "_c"], x[k + "_s"])
                ),
                axis=1,
            )
        )

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

    # Filter cohort: Remove hemorrhagic stroke
    data = data[data["ind_str"] != 2]

    # For Predictive Task
    data["3m_d"] = pd.to_datetime(data["3m_d"])
    data["dis_d"] = pd.to_datetime(data["dis_d"])
    data["cum_mve_1y_d"] = pd.to_datetime(data["cum_mve_1y_d"])

    data = data[(data["dis_d"] < data["3m_d"])]
    data = data[data["cum_mve_1y_d"].isna() | (data["dis_d"] < data["cum_mve_1y_d"])]
    data = data[
        (data["mrs3mo"] != "") & (data["mrs3mo"] != 9) & (data["cum_mve_1y"] != "")
    ]

    # Part 3. Merge cohort and metadata
    for col in data.columns:
        if col not in ["end1_memo", "end2_memo", "end3_memo", "edu", "psce1_12_ow"]:
            if col in metadata["SNUBH field name"].values:
                mapping_dict = metadata[metadata["SNUBH field name"] == col][
                    "mapping_dict"
                ].values[0]
                data[col] = data[col].map(lambda x: map_values(x, mapping_dict, col))
        kor_elems = [
            i
            for i in data[col].unique()
            if isinstance(i, str) and re.search(r"[\uAC00-\uD7A3]", i)
        ]
        if kor_elems:
            data[col] = data[col].replace(kor_to_eng_dict)

    data.fillna("", inplace=True)
    filtered = metadata[metadata["Row Name"].notna()]
    col_mapper = dict(zip(filtered["SNUBH field name"], filtered["Row Name"]))
    del col_mapper["edu1"]
    del col_mapper["edu2"]

    col_mapper["edu"] = "Education"
    col_mapper["psce1_12_ow"] = (
        "Potential Source of Cardioembolism/High Risk/Others Text"
    )

    col_mapper = {k: [v] for k, v in col_mapper.items()}

    # These columns include PII
    for i in ["trans_sub_1", "dis_sub_1", "ev1_hosw_3m", "ev1_hosw_1y", "comment"]:
        del col_mapper[i]

    # Part 5. Make Label
    data = data.rename(columns={"고유번호": "Unique ID"})

    target_cols = list(col_mapper.keys()) + ["Unique ID"]
    data = data[[i for i in data.columns if i in target_cols]]
    choices_df = get_choices(data, metadata)
    mi_matrix = get_mutual_info_matrix(data)

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
