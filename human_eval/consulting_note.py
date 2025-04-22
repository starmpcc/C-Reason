import os
import re

import gradio as gr
import pandas as pd

df = pd.read_csv("recommendation_20250312_df1_incl.csv")
df = df[df["incl"] == 1]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


def anonymize_korean_names(text):
    names = re.findall(
        r"(?:작성자|수신인|의뢰의사|지정의사)\s([\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]+)\s",
        text,
        re.MULTILINE,
    )
    for name in names:
        text = text.replace(name, "[이름]")
    return text


df["내용"] = df["내용"].apply(anonymize_korean_names)


def build_question(i, row):
    with gr.Accordion(label=f"Sample {i+1} (ID={row['id']})"):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    gr.Textbox(
                        value=row["내용"],
                        label="Consulting Note (Korean)",
                        lines=20,
                        max_lines=20,
                    )
                    gr.Textbox(
                        value=row["input"],
                        label=f"Consulting Note Model Input (English)",
                        lines=5,
                        max_lines=5,
                    )
                    gr.Textbox(
                        value=row["reference_answer"],
                        label=f"Reference Recommendation",
                        max_lines=20,
                    )
            with gr.Row():
                if row["shuffle"]:
                    rec_a = row["Trained"]
                    rec_b = row["phi_4"]
                else:
                    rec_a = row["phi_4"]
                    rec_b = row["Trained"]
                with gr.Column():
                    gr.Textbox(
                        rec_a,
                        label="Recommendation A",
                        lines=10,
                        max_lines=100,
                    )
                with gr.Column():
                    gr.Textbox(
                        rec_b,
                        label="Recommendation B",
                        lines=10,
                        max_lines=100,
                    )


def build_tab(df, idx):
    df = df.iloc[idx * 20 : idx * 20 + 20].reset_index(drop=True)
    with gr.Tab(str(idx + 1)):
        for i, row in df.iterrows():
            build_question(i, row)
    return


with gr.Blocks() as demo:
    for i in range(5):
        build_tab(df, i)

    demo.launch(auth=("admin", os.environ["PASSWORD"]), server_port=4444)
    demo.launch()
