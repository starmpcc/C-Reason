import argparse
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import get_value_choices


@dataclass
class DatasetConfig:
    num_choices: int
    difficulty_const: float


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def serialize(row, col_mapper, choices_df, question, target_col):
    dset_cfg = DatasetConfig(num_choices=5, difficulty_const=2.0)

    prompt = ""
    for k, v in row.items():
        if k in col_mapper:
            if k == target_col:
                continue
            if pd.notna(v) and v != "":
                prompt += f"{col_mapper[k][0]}: {str(v).strip()}\n"
    prompt = prompt.strip() + "\n" + question
    label, choices = get_value_choices(
        row[target_col], choices_df.loc[target_col], dset_cfg
    )
    label_idx = np.random.randint(len(choices) + 1)
    choices.insert(label_idx, label)

    for i in range(len(choices)):
        prompt += f"\n{chr(65+i)}. {choices[i]}"
    label_alpha = chr(65 + label_idx)
    return pd.Series(
        {
            "user_message": prompt,
            "label_0": label_alpha,
            "task": target_col,
            "choices_0": choices,
            "Unique ID": row["Unique ID"],
        }
    )


def main():
    args = parse_args()

    data = pd.read_csv(os.path.join(args.input_path, "valid.csv"))
    col_mapper = pickle.load(
        open(os.path.join(args.input_path, "col_mapper.pkl"), "rb")
    )
    choices_df = pd.read_pickle(os.path.join(args.input_path, "choices.pkl"))

    remove_cols = [i for i in data.columns if "3m" in i or "1y" in i]
    remove_cols += [
        "ava",
        "ava_no",
        "inform",
        "in_txt",
        "m_forget",
        "m_keep",
        "m_order",
        "k_good",
        "k_bad",
        "k_adv",
        "dose",
        "drug_stop",
        "drug_stop_w",
    ]
    remove_cols.remove("mrs3mo")
    remove_cols.remove("cum_mve_1y")
    data.drop(remove_cols, axis=1, inplace=True)

    mrs3mo_question = "Q. What will the mRS value be 3 months after admission?"
    mace_question = "Q. Would major adverse cardiovascular events (MACE) occur within one year after discharge?"
    mrs3mo = data.drop(columns=["cum_mve_1y"]).apply(
        lambda row: serialize(
            row,
            col_mapper,
            choices_df,
            mrs3mo_question,
            "mrs3mo",
        ),
        axis=1,
    )
    mace = data.drop(columns=["mrs3mo"]).apply(
        lambda row: serialize(
            row,
            col_mapper,
            choices_df,
            mace_question,
            "cum_mve_1y",
        ),
        axis=1,
    )
    df = pd.concat([mrs3mo, mace], ignore_index=True)
    dset = Dataset.from_pandas(df)
    dset = DatasetDict({"valid": dset})
    dset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
