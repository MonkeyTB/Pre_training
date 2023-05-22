from transformers import AutoModel, AutoTokenizer
import gradio as gr
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from chatglm.chatglm_model import ChatGlmModel
import argparse
import logging
import os
import mdtex2html

DEVICE = 'cuda'
DEVICE_ID = '0'
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
parser.add_argument('--model_name', default='../../THUDM/chatglm-6b', type=str, help='Transformers model or path')
parser.add_argument('--output_dir', default='./result/', type=str, help='Model output directory')
args = parser.parse_args()

model = ChatGlmModel(
    args.model_type, args.model_name,
    args={'use_lora': True, 'output_dir': args.output_dir},
    cuda_device=DEVICE_ID
)

MAX_TURNS = 1
MAX_BOXES = MAX_TURNS * 2


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(input, writebot, max_length, top_p, temperature, history):
    writebot.append((parse_text(input), ""))
    for response, history in model.stream_predict(input, [], max_length=max_length, top_p=top_p,
                                                  temperature=temperature, typical_p=0.2, repetition_penalty=1.2,
                                                  no_repeat_ngram_size=8):
        writebot[-1] = (parse_text(input), parse_text(response))
        yield writebot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">JD Write Bot</h1>""")

    writebot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 2048, value=600, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            no_repeat_ngram_size = gr.Slider(0, 20, value=8, step=1, label='no repeat ngram', interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, writebot, max_length, top_p, temperature, history], [writebot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[writebot, history], show_progress=True)

    gr.HTML("""<h5 align="right">问题反馈联系[壹叁]</h5>""")

demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=443)