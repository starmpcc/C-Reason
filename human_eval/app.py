import os
import pickle

import gradio as gr

pre_task_dict = pickle.load(open("human_eval/pre_task_dict_mort.pkl", "rb"))

tasks = pre_task_dict.keys()

task_dict = {}

for task, v in pre_task_dict.items():
    task_dict[task] = []
    before_dict = [
        dict(zip(v["before"].keys(), values)) for values in zip(*v["before"].values())
    ]
    after_dict = [
        dict(zip(v["after"].keys(), values)) for values in zip(*v["after"].values())
    ]
    for b, a, s in zip(before_dict, after_dict, v["shuffle"]):
        assert b["prompt"] == a["prompt"]

        prompt = b["prompt"].split("\n")
        num_choices = len(b["choices_0"])
        question = (
            "\n".join(prompt[-num_choices - 1 :])[:-66]
            + "\n\nCorrect Answer: "
            + b["label_0"]
        )
        prompt = "\n".join(prompt[: -num_choices - 1])

        task_item = {
            "prompt": prompt.split("user<|im_sep|>")[1].split("\nQ1.")[0],
            "question": question,
            "before": b["generated"].strip(),
            "after": a["generated"].strip(),
            "before_value": b["value_0"],
            "after_value": a["value_0"],
            "summary": b["summary"],
            "shuffle": s,
        }
        task_dict[task].append(task_item)

tasks = list(task_dict.keys())


def build_question(task, idx):
    with gr.Accordion(label=f"Sample {idx+1}"):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    gr.Textbox(
                        value=task_dict[task][idx]["prompt"],
                        label="Full Trajectory",
                        lines=10,
                        max_lines=10,
                    )
                    gr.Textbox(
                        value=task_dict[task][idx]["summary"],
                        label=f"Patient Summary",
                        lines=5,
                        max_lines=5,
                    )
                    gr.Textbox(
                        task_dict[task][idx]["question"],
                        label="Question & Answer",
                        lines=8,
                        max_lines=8,
                    )
            with gr.Row():
                if task_dict[task][idx]["shuffle"]:
                    res_a = task_dict[task][idx]["after"]
                    res_b = task_dict[task][idx]["before"]
                    ans_a = task_dict[task][idx]["after_value"]
                    ans_b = task_dict[task][idx]["before_value"]
                else:
                    res_a = task_dict[task][idx]["before"]
                    res_b = task_dict[task][idx]["after"]
                    ans_a = task_dict[task][idx]["before_value"]
                    ans_b = task_dict[task][idx]["after_value"]

                with gr.Column():
                    gr.Textbox(
                        res_a,
                        label="Reasoning A",
                        lines=10,
                        max_lines=100,
                    )
                    gr.Textbox(
                        ans_a,
                        label="Answer A",
                    )
                with gr.Column():
                    gr.Textbox(
                        res_b,
                        label="Reasoning B",
                        lines=10,
                        max_lines=100,
                    )
                    gr.Textbox(
                        ans_b,
                        label="Answer B",
                    )


def build_tab(task):
    with gr.Tab(task):
        for i in range(len(task_dict[task])):
            build_question(task, i)
    return


with gr.Blocks() as demo:
    for task in tasks:
        build_tab(task)

    demo.launch(auth=("admin", os.environ["PASSWORD"]), server_port=7888)
