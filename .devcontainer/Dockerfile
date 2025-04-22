# FROM mcr.microsoft.com/devcontainers/miniconda:0-3
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update -y \
    && apt-get install -y python3-pip sudo tmux git python-is-python3

RUN pip install accelerate==1.3.0 \
    transformers==4.48.3 \
    vllm==0.7.2 \
    peft==0.14.0 \
    trl==0.15.1 \
    liger-kernel==0.5.2 \
    bitsandbytes==0.45.2
RUN pip install h5py omegaconf gpustat ninja wandb h5pickle ipywidgets scipy pandas ipykernel matplotlib hydra-core nltk openai evaluate openpyxl scikit-learn xgboost gradio deepspeed pandarallel

ENV WANDB_PROJECT="LLM_Pred"
ENV WANDB__SERVICE_WAIT=300
ENV OMP_NUM_THREADS=8