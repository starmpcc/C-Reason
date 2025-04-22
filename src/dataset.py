import os
import pickle
import random
import re
from logging import getLogger

import numpy as np
import pandas as pd
from accelerate.utils import set_seed
from datasets import Dataset, disable_caching, load_from_disk
from scipy.stats import norm

logger = getLogger(__name__)

DATASET_REGISTRY = {}
LABEL_MAPPER_REGISTRY = {}
REGISTRY_SERIALIZER_REGISTRY = {}
disable_caching()


def user_message_to_template(sample, tokenizer, dset_cfg):
    chat = []
    if dset_cfg.get("sys_prompt"):
        chat.append({"role": "system", "content": dset_cfg.sys_prompt})
    chat.append({"role": "user", "content": sample["user_message"]})

    # Reasoning Models, do not need zero-shot CoT
    if tokenizer.name_or_path in [
        "Qwen/QwQ-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ]:
        templatized = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    else:
        chat.append({"role": "assistant", "content": "Let's think step by step."})
        templatized = tokenizer.apply_chat_template(
            chat, tokenize=False, continue_final_message=True
        )

    if dset_cfg.get("sys_prompt"):
        # Phi-3 just removes the sys_prompt
        if not dset_cfg.sys_prompt in templatized:
            chat = [
                {
                    "role": "user",
                    "content": dset_cfg.sys_prompt + "\n" + sample["user_message"],
                },
                {"role": "assistant", "content": "Let's think step by step."},
            ]
            templatized = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                continue_final_message=True,
            )

    return {"prompt": templatized}


def get_datasets(cfg, tokenizer):
    datasets = {}
    if len(cfg.dataset) == 0:
        raise ValueError("No dataset is specified")
    for k in cfg.dataset.keys():
        dset_cfg = cfg.dataset[k]

        if dset_cfg.cache_dir:
            dset = load_from_disk(dset_cfg.cache_dir)[cfg.launcher.split]
            if "prompt" not in dset.column_names:
                dset = dset.map(
                    user_message_to_template,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "dset_cfg": dset_cfg,
                    },
                    remove_columns="user_message",
                    num_proc=32,
                )
            if cfg.launcher.split == "train":
                dset = dset.select(range(dset_cfg.num_train_samples))

            if cfg.debug:
                dset = dset.select(range(cfg.debug_samples))

            datasets[k] = dset
            continue

        # Each dataset should have same contents regardless of the order
        seed = 42
        set_seed(seed)

        mapper = DATASET_REGISTRY[dset_cfg.mapper_name]
        split = cfg.launcher.split

        if cfg.debug:
            data_length = cfg.debug_samples
        elif split == "train":
            data_length = dset_cfg.num_train_samples
        elif split == "valid":
            data_length = dset_cfg.num_eval_samples

        if dset_cfg.mapper_name == "registry":
            dataset = mapper(dset_cfg, tokenizer, split, data_length)
            dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(min(data_length, len(dataset))))

        else:
            dataset = load_from_disk(dset_cfg.data_path)
            if split is not None:
                dataset = dataset.filter(
                    lambda x: x["split_2020"] == split, num_proc=32
                )

            mapper_kwargs = {"dset_cfg": dset_cfg, "seed": seed}

            dataset = dataset.map(
                mapper,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=32,
                fn_kwargs=mapper_kwargs,
            )

            if "user_message" in dataset.column_names:
                dataset = dataset.map(
                    user_message_to_template,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "dset_cfg": dset_cfg,
                    },
                    remove_columns="user_message",
                    num_proc=32,
                )
            dataset = dataset.shuffle(seed=seed)

            # If too long -> filter by length
            if len(dataset) > data_length:
                dataset = dataset.select(range(min(len(dataset), data_length * 4)))
                for col, limit in zip(
                    ["prompt", "text"],
                    [
                        dset_cfg.max_prompt_tokens,
                        dset_cfg.max_text_tokens,
                    ],
                ):
                    if col in dataset.column_names:
                        dataset = dataset.map(
                            lambda x: {
                                "len": [len(i) for i in tokenizer(x[col]).input_ids]
                            },
                            batched=True,
                            num_proc=32,
                        )
                        dataset = dataset.filter(
                            lambda x: x["len"] < limit,
                            num_proc=32,
                        ).remove_columns("len")

            dataset = dataset.select(range(min(data_length, len(dataset))))

        datasets[k] = dataset
    return datasets


def register_dataset(name):
    def register_dataset(func):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate Dataset ({})".format(name))
        DATASET_REGISTRY[name] = func

        return func

    return register_dataset


def register_label_mapper(name):
    def register_label_mapper(func):
        if name in LABEL_MAPPER_REGISTRY:
            raise ValueError("Cannot register duplicate label_mapper ({})".format(name))
        LABEL_MAPPER_REGISTRY[name] = func

        return func

    return register_label_mapper


def get_value_choices(label, col, dset_cfg):
    try:
        label = str(float(label))
        is_float = True
    except:
        is_float = False

    if col["FLOAT_RATIO"] == 0 and len(col["VOCAB"]) - 1 <= dset_cfg.num_choices - 1:
        choices = [i for i in col["VOCAB"] if i != label]
    else:
        num_cat_choices = np.random.binomial(
            dset_cfg.num_choices - 1, 1 - col["FLOAT_RATIO"]
        )

        num_cat_choices = min(num_cat_choices, len(col["VOCAB"]) - int(not is_float))
        num_numeric_choices = dset_cfg.num_choices - 1 - num_cat_choices

        choices = []
        if num_cat_choices > 0:
            vocab, weights = zip(
                *[
                    (col["VOCAB"][i], col["WEIGHTS"][i])
                    for i in range(len(col["VOCAB"]))
                    if col["VOCAB"][i] != str(label)
                ]
            )

            weights = [i / sum(weights) for i in weights]
            choices += np.random.choice(
                vocab,
                size=num_cat_choices,
                p=weights,
                replace=False,
            ).tolist()

        if num_numeric_choices > 0:
            if is_float:
                _label = float(label)
            else:
                _label = np.random.choice(
                    col["GM"]["means"], size=1, p=col["GM"]["weights"]
                )[0]

            scores = [
                norm.pdf(_label, col["GM"]["means"][i], col["GM"]["stds"][i])
                * col["GM"]["weights"][i]
                for i in range(3)
            ]

            if np.sum(scores) == 0:
                candidates = sorted(
                    list(set(col["ALL_VOCAB"]) - {label} - set(choices))
                )
                choices += random.sample(candidates, num_numeric_choices)
            else:
                selected_gaussian = np.random.choice(
                    range(3), size=1, p=scores / np.sum(scores)
                )[0]

                # If gaussian has sharp peak -> use another method
                if col["GM"]["stds"][selected_gaussian] < 0.1:
                    candidates = sorted(
                        list(set(col["ALL_VOCAB"]) - {label} - set(choices))
                    )
                    choices += random.sample(candidates, num_numeric_choices)
                else:
                    order_of_label = np.random.randint(num_numeric_choices + 1)

                    numeric_choices = [
                        _label
                        + dset_cfg.difficulty_const
                        * (i - order_of_label)
                        * col["GM"]["stds"][selected_gaussian]
                        for i in range(num_numeric_choices + 1)
                        if i != order_of_label
                    ]

                    float_items = col["ALL_VOCAB"] - set(col["VOCAB"])

                    # First, remove if corresponding values are mostly positive
                    pos_items = [i for i in float_items if float(i) >= 0]
                    if len(pos_items) > len(float_items) * 0.95:
                        numeric_choices = [i for i in numeric_choices if i >= 0]

                    # Second, round if corresponding values are mostly integers
                    int_items = [i for i in float_items if float(i) // 1 == float(i)]
                    if len(int_items) > len(float_items) * 0.9:
                        numeric_choices = [str(round(i)) for i in numeric_choices]
                    else:
                        numeric_choices = [str(round(i, 1)) for i in numeric_choices]

                    choices += numeric_choices

    assert label != "NA"
    if "nan" in choices:
        choices.remove("nan")

    # Convert to int if float is actually int
    choices = [re.sub(r"(?<=\d)\.0+$|(\.\d*?[1-9])0+$", r"\1", i) for i in choices]
    label = re.sub(r"(?<=\d)\.0+$|(\.\d*?[1-9])0+$", r"\1", label)

    # Round can make duplicates
    choices = list(set(choices) - {label})

    # set() operation shuffle order randomly
    choices = sorted(choices)

    return label, choices


@register_dataset("registry")
def registry_mapper(dset_cfg, tokenizer, split, data_length, **kwargs):
    df = pd.read_csv(os.path.join(dset_cfg.data_path, split + ".csv"))
    choices_df = pd.read_pickle(os.path.join(dset_cfg.data_path, "choices.pkl"))
    mi_matrix = pd.read_pickle(os.path.join(dset_cfg.data_path, "mi_matrix.pkl"))
    with open(os.path.join(dset_cfg.data_path, "col_mapper.pkl"), "rb") as f:
        col_mapper = pickle.load(f)

    if dset_cfg.all_victims:
        mappeds = []
        for col in df.columns:
            if col in choices_df.index:
                mapped = df.apply(
                    registry_processor,
                    axis=1,
                    choices_df=choices_df,
                    mi_matrix=mi_matrix,
                    col_mapper=col_mapper,
                    dset_cfg=dset_cfg,
                    fixed_victims=[col],
                )
                mappeds.append(mapped)
        mapped = pd.concat(mappeds)
    elif len(df) < data_length:
        assert dset_cfg.get("fixed_victims") == None
        len_data = 0
        mappeds = []
        while len_data < data_length * 2:  # To dedup
            mapped = df.apply(
                registry_processor,
                axis=1,
                choices_df=choices_df,
                mi_matrix=mi_matrix,
                col_mapper=col_mapper,
                dset_cfg=dset_cfg,
                fixed_victims=None,
            )
            mappeds.append(mapped)
            len_data += len(mapped)

        mapped = pd.concat(mappeds)
        mapped = mapped.drop_duplicates(["user_message"])

    else:
        mapped = df.apply(
            registry_processor,
            axis=1,
            choices_df=choices_df,
            mi_matrix=mi_matrix,
            col_mapper=col_mapper,
            dset_cfg=dset_cfg,
            fixed_victims=dset_cfg.get("fixed_victims"),
        )

    # Skip if nothing returned
    mapped = mapped[mapped["user_message"].notna()]

    dset = Dataset.from_pandas(mapped)

    dset = dset.map(
        user_message_to_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "dset_cfg": dset_cfg,
        },
        remove_columns="user_message",
    )
    dset = dset.add_column("split_2020", [split] * len(dset))
    return dset


def register_registry_serializer(name):
    def register_registry_serializer(func):
        if name in REGISTRY_SERIALIZER_REGISTRY:
            raise ValueError(
                "Cannot register duplicate registry_serializer ({})".format(name)
            )
        REGISTRY_SERIALIZER_REGISTRY[name] = func

        return func

    return register_registry_serializer


def registry_processor(row, choices_df, mi_matrix, col_mapper, dset_cfg, fixed_victims):
    victims = []
    high_mis = []
    prompt = ""
    res_dict = {}
    while len(victims) < dset_cfg.num_questions:
        victim_candidates = [
            k
            for k, v in row.items()
            if pd.notna(v)
            and v != ""
            and v != "NA"
            and k not in ["Unique ID", *victims, *high_mis]
        ]

        if fixed_victims is not None:
            victim = fixed_victims[len(victims)]
            if victim not in victim_candidates or victim not in choices_df.index:
                return pd.Series()

        else:
            victim = random.choice(victim_candidates)
            if victim not in choices_df.index:
                continue

        label, choices = get_value_choices(
            row[victim], choices_df.loc[victim], dset_cfg
        )
        if len(choices) == 0:
            if fixed_victims is not None:
                return pd.Series()
            else:
                continue

        victims.append(victim)

        if victim in mi_matrix.columns:
            high_mi = mi_matrix[victim][
                mi_matrix[victim] > dset_cfg.mi_cutoff
            ].index.tolist()
            high_mis += high_mi

        random.shuffle(choices)
        label_idx = np.random.randint(len(choices) + 1)
        choices.insert(label_idx, label)
        if victim in col_mapper:
            prompt += "\n" + dset_cfg.question.format(len(victims), *col_mapper[victim])
        else:
            prompt += "\n" + dset_cfg.question

        for j in range(len(choices)):
            prompt += f"\n{chr(65+j)}. {choices[j]}"
        label_alpha = chr(65 + label_idx)

        res_dict[f"label_{len(victims)-1}"] = label_alpha
        res_dict[f"choices_{len(victims)-1}"] = choices

        if len(victims) == dset_cfg.num_questions:
            patient_info = REGISTRY_SERIALIZER_REGISTRY[dset_cfg.serializer_name](
                row, victims, col_mapper, high_mis
            )
            if patient_info == "":
                if fixed_victims is not None:
                    return pd.Series()
                victims = []
                high_mis = []
                prompt = ""
                res_dict = {}
                continue
            else:
                prompt = patient_info + prompt

    res_dict.update(
        {
            "Unique ID": row["Unique ID"],
            "user_message": prompt,
            "victims": victims,
        }
    )

    return pd.Series(res_dict)


@register_registry_serializer("sepsis_registry")
def sepsis_registry_serializer(row, victims, col_mapper, high_mis):
    data = ""
    current_header = ""
    for i in range(len(row)):  # For serialized, token_len
        if row.index[i] not in col_mapper:
            continue
        elif row.index[i] in high_mis:
            continue
        new_name, header = col_mapper[row.index[i]]
        value = row.values[i]
        if pd.notna(row.values[i]) and row.values[i] != "":
            if current_header != header:
                data += "\n" + header + "\n"
                current_header = header
            if row.index[i] in victims:
                value = "[MASK]"
            data += f"{new_name.strip()}: {value}\n"

    # If hospital info exists -> move to first
    if "[Derived variable/Hospital factor]" in data:
        body, hosp = data.split("[Derived variable/Hospital factor]")
        data = (
            "[Baseline Characteristics/Hospital Information]\n"
            + hosp.strip()
            + "\n\n"
            + body.strip()
        )

    return data.strip()


@register_registry_serializer("stroke_registry")
def stroke_registry_serializer(row, victims, col_mapper, high_mis):
    data = ""
    for k, v in row.items():
        if k in col_mapper:
            if pd.notna(v) and v != "":
                if k in victims:
                    v = "[MASK]"
                data += f"{col_mapper[k][0]}: {str(v).strip()}\n"

    return data.strip()


@register_registry_serializer("aki_registry")
def aki_registry_serializer(row, victims, col_mapper, high_mis):
    data = ""
    current_header = ""
    notna_flag = True
    for i in range(len(row)):  # For serialized, token_len
        if row.index[i] not in col_mapper:
            continue
        elif row.index[i] in high_mis:
            continue
        new_name, header = col_mapper[row.index[i]]
        value = row.values[i]

        if current_header != header:
            if notna_flag == False:
                data = data.strip(current_header).strip()
                if "[MASK]" not in data:
                    return ""
                else:
                    return data
            data += "\n" + header + "\n"
            current_header = header
            notna_flag = False

        if pd.notna(row.values[i]) and row.values[i] != "":
            if row.index[i] in victims:
                value = "[MASK]"
            data += f"{new_name.strip()}: {value}\n"
            notna_flag = True

    return data.strip()
