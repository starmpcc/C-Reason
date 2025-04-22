import argparse
from logging import getLogger

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--model_path", type=str)
    pass


def map_category(x):
    if "투여할 제한항생제" in x:
        return "restricted_antibiotics"
    elif "[오더발행]" in x:
        return "blood_culture"
    else:
        return "regular"


def make_translate(notes, model, tokenizer):
    messages = [
        [
            {
                "role": "user",
                "content": f"Translate the whole conversation into English. All entities must be translated. Do not add comment after translation.\n{note}",
            },
            {
                "role": "assistant",
                "content": "Here is the translation of the conversation:\n\n",
            },
        ]
        for note in notes
    ]
    templetizeds = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    templetizeds = [t[:-10] for t in templetizeds]
    sp = SamplingParams(temperature=0.0, max_tokens=5664)
    out = model.generate(templetizeds, sp)
    return [o.outputs[0].text.strip() for o in out]


def make_anonymize(notes, model, tokenizer):
    messages = [
        [
            {
                "role": "user",
                "content": f"Find all human name in the below conversation. Only list up name.\n{note}",
            },
            {
                "role": "assistant",
                "content": "The names mentioned in the conversation are:\n\n1. ",
            },
        ]
        for note in notes
    ]
    templetizeds = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    templetizeds = [t[:-10] for t in templetizeds]
    sp = SamplingParams(temperature=0.0, max_tokens=5664)
    out = model.generate(templetizeds, sp)

    anonymized_notes = []
    for o, note in zip(out, notes):
        names = o.outputs[0].text.strip().split("\n")
        names = [i.split(". ")[-1] for i in names]
        for name in names:
            note = note.replace(name, "[NAME]")
        anonymized_notes.append(note)
    return anonymized_notes


def split_section(notes, model, tokenizer):
    messages = [
        [
            {
                "role": "user",
                "content": """The given conversation consists of two messages: 1) referral request and 2) response.
                The response consists of two parts: 2-1) patient information and 2-2) assessment/opinion/recommendation.
                Copy the first section header of 2-2) assessment/opinion/recommendation part from the response (e.g., "option or recommendation\n", "assessment:\n", "**Assessment**\n", "Assessment\n", "opinion\n").
                If no section header, just copy the first line of the 2-2 part without comment.
                Do not add any extra text after the answer.
                """
                + note,
            },
            {
                "role": "assistant",
                "content": "",
            },
        ]
        for note in notes
    ]
    templetizeds = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    templetizeds = [t[:-10] for t in templetizeds]
    sp = SamplingParams(temperature=0.0, max_tokens=5664)
    out = model.generate(templetizeds, sp)

    splitteds = []
    for o, note in zip(out, notes):
        header = o.outputs[0].text.strip()
        cleaned_header = header.replace("*", "").strip()

        splitted = None
        for hd in [header, cleaned_header]:
            if note.count(hd) == 1:
                prev, assessment = note.split(hd)
                splitted = prev + "*" * 50 + "\n" + hd + "\n" + assessment
                break
            elif note.count(hd) == 2:
                prev, prev_2, assessment = note.split(hd)
                prev = prev + "\n" + hd + "\n" + prev_2
                splitted = prev + "*" * 50 + "\n" + hd + "\n" + assessment
                break

        if splitted == None:
            print(header)
            print(note.count(header))
            print("*" * 100)
            print(note)
            print("*" * 100)
        splitteds.append(splitted)

    return splitteds


def main():
    args = parse_args()

    # 5664: max len for FP8 70B Llama on H100
    model = LLM(args.model_path, max_model_len=5664)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    df = pd.read_csv(args.input)

    df["category"] = df["내용"].map(map_category)
    df["num_tokens"] = df["내용"].map(lambda x: len(tokenizer.encode(x)))

    df["translated"] = make_translate(df["내용"], model, tokenizer)
    df.to_csv("translated.csv", index=False)

    df["anonymized"] = make_anonymize(df["translated"], model, tokenizer)
    df.to_csv("anonymized.csv", index=False)

    df["splitted"] = split_section(df["anonymized"], model, tokenizer)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
