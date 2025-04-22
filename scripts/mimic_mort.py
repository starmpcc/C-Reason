import argparse
import os
import random

import pandas as pd
from datasets import Dataset, DatasetDict

feature_name_mapping = {
    "o:gender": "Gender",
    "o:mechvent": "Mechanical Ventilation",
    "o:max_dose_vaso": "Maximum Vasopressor Dose",
    "o:re_admission": "Readmission",
    "o:age": "Age",
    "o:Weight_kg": "Weight (kg)",
    "o:GCS": "GCS",
    "o:HR": "HR",
    "o:SysBP": "Systolic BP",
    "o:MeanBP": "Mean BP",
    "o:DiaBP": "Diastolic BP",
    "o:RR": "RR",
    "o:Temp_C": "Temperature (°C)",
    "o:FiO2_1": "FiO2",
    "o:Potassium": "Potassium",
    "o:Sodium": "Sodium",
    "o:Chloride": "Chloride",
    "o:Glucose": "Glucose",
    "o:Magnesium": "Magnesium",
    "o:Calcium": "Calcium",
    "o:Hb": "Hb",
    "o:WBC_count": "WBC Count",
    "o:Platelets_count": "Platelet Count",
    "o:PTT": "PTT",
    "o:PT": "PT",
    "o:Arterial_pH": "Arterial pH",
    "o:paO2": "paO2",
    "o:paCO2": "paCO2",
    "o:Arterial_BE": "Arterial BE",
    "o:HCO3": "HCO3",
    "o:Arterial_lactate": "Arterial Lactate",
    "o:SOFA": "SOFA",
    "o:SIRS": "SIRS",
    "o:Shock_Index": "Shock Index",
    "o:PaO2_FiO2": "PaO2/FiO2",
    "o:cumulated_balance": "Cumulative Fluid Balance",
    "o:SpO2": "SpO2",
    "o:BUN": "BUN",
    "o:Creatinine": "Creatinine",
    "o:SGOT": "SGOT",
    "o:SGPT": "SGPT",
    "o:Total_bili": "Total Bilirubin",
    "o:INR": "INR",
    "o:input_total": "Total Fluid Input",
    "o:input_4hourly": "4-Hour Fluid Input",
    "o:output_total": "Total Fluid Output",
    "o:output_4hourly": "4-Hour Fluid Output",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--mimic_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def serialize(df):
    results = "[Patient Information]\n"
    for k in ["Gender", "Age", "Readmission"]:
        results += f"{k}: {df.iloc[0][k]}\n"

    for _, row in df.iterrows():
        time = int((row["m:charttime"] - row["m:presumed_onset"]) // 3600)
        time = str(time) if time < 0 else "+" + str(time)
        results += f"\n[Time: Onset{time}h]\n"
        for k, v in row.items():
            if k in feature_name_mapping.values() and k not in [
                "Gender",
                "Age",
                "Readmission",
            ]:
                if isinstance(v, str):
                    results += f"{k}: {v}\n"
                elif int(v) == v:
                    results += f"{k}: {int(v)}\n"
                else:
                    results += f"{k}: {v:.3f}\n"

    user_message = (
        results + f"Q. Is the patient likely to die in the hospital?\nA. Yes\nB. No"
    )

    label = "A" if bool(df.iloc[0]["IN_HOSP_MORT"]) else "B"

    return pd.Series(
        {
            "icustayid": int(df.iloc[0]["m:icustayid"]),
            "user_message": user_message,
            "label_0": label,
            "task": "mortality",
            "choices_0": ["Yes", "No"],
        }
    )


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)

    icustays = pd.read_csv(os.path.join(args.mimic_path, "ICUSTAYS.csv"))
    admissions = pd.read_csv(os.path.join(args.mimic_path, "ADMISSIONS.csv"))
    patients = pd.read_csv(os.path.join(args.mimic_path, "PATIENTS.csv"))

    icustays = icustays.merge(
        admissions[["HADM_ID", "DISCHTIME", "DEATHTIME"]], on="HADM_ID"
    )
    icustays = icustays.merge(patients[["SUBJECT_ID", "DOD"]], on="SUBJECT_ID")

    icustays["DOD"] = pd.to_datetime(icustays["DOD"])
    icustays["DISCHTIME"] = pd.to_datetime(icustays["DISCHTIME"])

    icustays["IN_HOSP_MORT"] = (
        icustays["DOD"] - icustays["DISCHTIME"]
    ) <= pd.Timedelta(1, unit="D")

    df = df.merge(
        icustays[
            [
                "ICUSTAY_ID",
                "IN_HOSP_MORT",
            ]
        ],
        left_on="m:icustayid",
        right_on="ICUSTAY_ID",
    ).drop(columns=["ICUSTAY_ID"])

    df["o:age"] = round(df["o:age"] / 365.25)
    df["o:gender"] = df["o:gender"].map(lambda x: "Female" if x else "Male")
    df["o:mechvent"] = df["o:mechvent"].map(lambda x: "Yes" if x else "No")
    df["o:re_admission"] = df["o:re_admission"].map(lambda x: "Yes" if x else "No")

    df = df.rename(columns=feature_name_mapping)

    serialized = df.groupby("m:icustayid").apply(serialize)
    serialized.reset_index(drop=True, inplace=True)

    random.seed(42)

    sample_ids = random.sample(list(serialized["icustayid"].unique()), 1000)
    serialized = serialized[serialized["icustayid"].isin(sample_ids)]

    dset = Dataset.from_pandas(serialized)
    dset = DatasetDict({"valid": dset})
    dset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
