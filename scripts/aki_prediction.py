import argparse
import os
import random
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=32, progress_bar=True)
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--original_path", type=str)

    parser.add_argument("--output_path", type=str)

    return parser.parse_args()


def serialize(row, col_mapper):
    labels, prompts, choices = [], [], []
    aki_seq = [row[f"d{d}_{t}_aki"] for d in range(1, 9) for t in range(1, 4)]
    aki_seq = [{"Yes": 1, "No": 0}.get(i, i) for i in aki_seq]

    data = ""
    current_header = ""
    notna_flag = True
    days, times = 0, 0
    for i in range(len(row)):  # For serialized, token_len
        if row.index[i] not in col_mapper:
            continue
        elif "aki" in row.index[i]:
            continue
        new_name, header = col_mapper[row.index[i]]
        value = row.values[i]

        if current_header != header:
            if notna_flag == False:
                return labels, prompts, choices
            # Timewise
            if re.match(r"\[Day \d \d", current_header):
                if times != 2 and not (days == 0 and times == 0):
                    label = (
                        sum(aki_seq[(days) * 3 + times : (days + 2) * 3 + times]) >= 1
                    )
                    if random.random() > 0.5:
                        label = "A" if label else "B"
                        prompt = (
                            data
                            + "\nQ. Would AKI occur in the next 48 hours?\nA.yes\nB.no"
                        )
                        choice = ["yes", "no"]
                    else:
                        label = "B" if label else "A"
                        prompt = (
                            data
                            + "\nQ. Would AKI occur in the next 48 hours?\nA.no\nB.yes"
                        )
                        choice = ["no", "yes"]
                    labels.append(label)
                    prompts.append(prompt)
                    choices.append(choice)
                times += 1
                times %= 3

            # Daily
            elif re.match(r"\[Day \d\]", current_header):
                label = sum(aki_seq[(days) * 3 + 2 : (days + 2) * 3 + 2]) >= 1
                if random.random() > 0.5:
                    label = "A" if label else "B"
                    prompt = (
                        data + "\nQ. Would AKI occur in the next 48 hours?\nA.yes\nB.no"
                    )
                    choice = ["yes", "no"]
                else:
                    label = "B" if label else "A"
                    prompt = (
                        data + "\nQ. Would AKI occur in the next 48 hours?\nA.no\nB.yes"
                    )
                    choice = ["no", "yes"]
                labels.append(label)
                prompts.append(prompt)
                choices.append(choice)
                days += 1

            if days == 8:
                break
            data += "\n" + header + "\n"
            current_header = header
            notna_flag = False

        if pd.notna(row.values[i]) and row.values[i] != "":
            data += f"{new_name.strip()}: {value}\n"
            notna_flag = True

    return labels, prompts, choices


def main():
    args = parse_args()
    data = pd.read_csv(os.path.join(args.data_path, "valid.csv"))
    orig = pd.read_csv(os.path.join(args.original_path, "data_processed.csv"))
    missing = pd.read_csv(
        os.path.join(args.original_path, "data_missing_processed.csv")
    )

    orig = orig[[f"d8_{i}_aki" for i in range(1, 4)] + ["Study_ID"]]
    missing = missing[[f"d8_{i}_aki" for i in range(1, 4)] + ["Study_ID"]]
    orig = orig.mask(missing == 0, np.nan)

    data = data.merge(orig, left_on="Unique ID", right_on="Study_ID", how="left")

    col_mapper = pd.read_pickle(
        os.path.join(args.data_path, "col_mapper.pkl"),
    )

    df = data.apply(
        lambda x: serialize(x, col_mapper),
        axis=1,
        result_type="expand",
    )
    df.columns = ["label_0", "user_message", "choices_0"]
    df["Unique ID"] = data["Study_ID"]
    df = df.explode(["label_0", "user_message", "choices_0"])
    df["user_message"] = df["user_message"].str.strip()

    # 1 sample per 1 pat
    df = df.dropna()
    ids = random.sample(df["Unique ID"].unique().tolist(), 1000)
    df = df[df["Unique ID"].isin(ids)]
    df = df.groupby("Unique ID").sample(n=1, random_state=42).reset_index(drop=True)

    dset = Dataset.from_pandas(df)
    dset = DatasetDict({"valid": dset})
    dset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
