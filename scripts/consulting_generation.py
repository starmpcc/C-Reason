import argparse

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

instruction = "Given the consult request with the patient's clinical background and the partial response from the infectious disease expert, provide a detailed antibiotic recommendation with thorough clinical reasoning. Emphasize antibiotic stewardship in your decision-making. Note that the sensitivity results only include a subset of antibiotics and are used to assess antibiotic resistance—not to strictly dictate the antibiotic choice. You may recommend antibiotics that are not listed in the sensitivity results if they are clinically justified.\n\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    return parser.parse_args()


def make_model_input(input, tokenizer):
    input = input.split("*" * 50)[0]
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instruction + input},
            {"role": "assistant", "content": "Let's think step by step."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    args = parse_args()
    model = LLM(args.model_path)
    sp = SamplingParams(max_tokens=16 * 1024, temperature=0.0)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")

    df = pd.read_csv(args.input)
    model_inputs = [make_model_input(i, tokenizer) for i in df["splitted"]]

    out = model.generate(model_inputs, sp)
    df[args.model_path] = [i.outputs[0].text for i in out]
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
