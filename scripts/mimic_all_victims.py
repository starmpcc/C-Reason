import argparse
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict
from pandarallel import pandarallel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sepsis_preprocessing import get_choices

from src.dataset import get_value_choices

set_seed(42)
pandarallel.initialize(nb_workers=32)


@dataclass
class DatasetConfig:
    num_choices: int
    difficulty_const: float


feature_name_mapping = {
    "o:gender": "Gender",
    "o:mechvent": "Mechanical Ventilation",
    "o:max_dose_vaso": "Maximum Vasopressor Dose over Recent 4h (mcg/kg/min of norepinephrine equivalent)",
    "o:re_admission": "Readmission",
    "o:age": "Age",
    "o:Weight_kg": "Weight (kg)",
    "o:GCS": "GCS",
    "o:HR": "HR (bpm)",
    "o:SysBP": "Systolic BP (mmHg)",
    "o:MeanBP": "Mean BP (mmHg)",
    "o:DiaBP": "Diastolic BP (mmHg)",
    "o:RR": "RR (breaths/min)",
    "o:Temp_C": "Temperature (°C)",
    "o:FiO2_1": "FiO2",
    "o:Potassium": "Potassium (mEq/L)",
    "o:Sodium": "Sodium (mEq/L)",
    "o:Chloride": "Chloride (mEq/L)",
    "o:Glucose": "Glucose (mg/dL)",
    "o:Magnesium": "Magnesium (mg/dL)",
    "o:Calcium": "Calcium (mg/dL)",
    "o:Hb": "Hb (g/dL)",
    "o:WBC_count": "WBC Count (K/ul)",
    "o:Platelets_count": "Platelet Count (K/ul)",
    "o:PTT": "PTT (sec)",
    "o:PT": "PT (sec)",
    "o:Arterial_pH": "Arterial pH",
    "o:paO2": "paO2 (mmHg)",
    "o:paCO2": "paCO2 (mmHg)",
    "o:Arterial_BE": "Arterial BE (mEq/L)",
    "o:HCO3": "HCO3 (mEq/L)",
    "o:Arterial_lactate": "Arterial Lactate (mmol/L)",
    "o:SOFA": "SOFA",
    "o:SIRS": "SIRS",
    "o:Shock_Index": "Shock Index",
    "o:PaO2_FiO2": "PaO2/FiO2",
    "o:cumulated_balance": "Cumulative Fluid Balance",
    "o:SpO2": "SpO2 (%)",
    "o:BUN": "BUN (mg/dL)",
    "o:Creatinine": "Creatinine (mg/dL)",
    "o:SGOT": "SGOT (U/L)",
    "o:SGPT": "SGPT (U/L)",
    "o:Total_bili": "Total Bilirubin (mg/dL)",
    "o:INR": "INR",
    "o:input_total": "Total Fluid Input (mL)",
    "o:input_4hourly": "4-Hour Fluid Input (mL)",
    "o:output_total": "Total Fluid Output (mL)",
    "o:output_4hourly": "4-Hour Fluid Output (mL)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    dset_cfg = DatasetConfig(num_choices=5, difficulty_const=2.0)

    df = pd.read_csv(
        args.data_path,
    )

    df["o:age"] = round(df["o:age"] / 365.25)
    df["o:gender"] = df["o:gender"].map(lambda x: "Female" if x else "Male")
    df["o:mechvent"] = df["o:mechvent"].map(lambda x: "Yes" if x else "No")
    df["o:re_admission"] = df["o:re_admission"].map(lambda x: "Yes" if x else "No")
    df["o:GCS"] = df["o:GCS"].map(lambda x: round(x))  # Extrapolated value -> Round!

    df = df.rename(columns=feature_name_mapping)

    choices_df = get_choices(df)

    question_template = (
        "\nQ. What will the {} value likely be 24 hours after the last records?"
    )

    filter_ids = (
        df.groupby("m:icustayid")
        .filter(
            lambda x: (x["m:charttime"].max() - x["m:charttime"].min() >= 48 * 3600)
            & (x["m:charttime"].max() - 24 * 3600 in x["m:charttime"].values)
        )["m:icustayid"]
        .unique()
    )

    filter_ids = random.sample(list(filter_ids), 1000)
    df = df[df["m:icustayid"].isin(filter_ids)]

    def serialize(df):
        set_seed(42)

        inputs = "[Patient Information]\n"
        for k in ["Gender", "Age", "Readmission"]:
            inputs += f"{k}: {df.iloc[0][k]}\n"

        for i, row in df.iterrows():
            if (df["m:charttime"].max() - row["m:charttime"]) < 24 * 3600:
                continue
            time = int((row["m:charttime"] - row["m:presumed_onset"]) // 3600)
            time = str(time) if time < 0 else "+" + str(time)
            inputs += f"\n[Time: Onset{time}h]\n"
            for k, v in row.items():
                if k in feature_name_mapping.values() and k not in [
                    "Gender",
                    "Age",
                    "Readmission",
                ]:
                    if isinstance(v, str):
                        inputs += f"{k}: {v}\n"
                    elif int(v) == v:
                        inputs += f"{k}: {int(v)}\n"
                    else:
                        inputs += f"{k}: {v:.3f}\n"

        prompts, victims, labels, choicess = [], [], [], []
        for target in df.columns:
            if target in ["Gender", "Age", "Readmission", "Mechanical Ventilation"]:
                continue
            if target in feature_name_mapping.values():
                question = question_template.format(target)
                label, choices = get_value_choices(
                    df.iloc[-1][target], choices_df.loc[target], dset_cfg
                )
                label_idx = np.random.randint(len(choices) + 1)
                choices.insert(label_idx, label)

                prompt = inputs + question
                for i in range(len(choices)):
                    prompt += f"\n{chr(65+i)}. {choices[i]}"
                label_alpha = chr(65 + label_idx)

                prompts.append(prompt)
                victims.append(target)
                labels.append(label_alpha)
                choicess.append(choices)

        return pd.DataFrame(
            {
                "icustayid": [int(df.iloc[0]["m:icustayid"])] * len(prompts),
                "user_message": prompts,
                "label_0": labels,
                "task": victims,
                "choices_0": choicess,
            }
        )

    serialized = df.groupby("m:icustayid").parallel_apply(serialize)
    serialized = serialized.dropna()
    serialized.reset_index(drop=True, inplace=True)

    serialized = serialized[serialized["choices_0"].str.len() >= 2]

    dset = Dataset.from_pandas(serialized)
    dset = DatasetDict({"valid": dset})
    dset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
