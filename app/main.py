# import os
# from typing import Optional
import librosa
import pathlib
from loguru import logger
import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
from models import SynthesizerTrn
from text import text_to_sequence
from utils import time_it
from mel_processing import spectrogram_torch
# import argparse
import re
from config import Config

brackets = ['（', '[', '『', '「', '【', ")", "】", "]", "』", "」", "）"]
pattern = re.compile('|'.join(map(re.escape, brackets)))


# def text_cleanner(text: str):
#     # text = re.sub(pattern, ' ', text)
#     text = pattern.sub(' ', text)
#     return text.strip()


if __name__ == '__main__':
    # if not os.getenv("INIT"):
    # setup_model()
    Config.init()
    app = gr.Blocks()

    with app:
        gr.Markdown(
            "a demo of web service of vits, thanks to @CjangCjengh, copy from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts)")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Column():
                    tts_input1 = gr.TextArea(
                        label="TTS_text", value="わたしの趣味はたくさんあります。でも、一番好きな事は写真をとることです。")
                    tts_input2 = gr.Dropdown(
                        label="Speaker", choices=Config.speaker_choices, type="index", value=Config.speaker_choices[0])
                    tts_input3 = gr.Slider(
                        label="Speed", value=1, minimum=0.2, maximum=3, step=0.1)

                    tts_submit = gr.Button("Generate", variant="primary")
                    tts_output1 = gr.Textbox(label="Output Message")
                    tts_output2 = gr.Audio(label="Output Audio")

            with gr.TabItem("Voice Conversion"):
                gr.Markdown(
                    "***please upload wav file, other format is not supported now.***")
                with gr.Column():
                    vc_input1 = gr.Dropdown(label="Original Speaker", choices=Config.hps.speakers, type="index",
                                            value=Config.hps.speakers[0])
                    vc_input2 = gr.Dropdown(label="Target Speaker", choices=Config.hps.speakers, type="index",
                                            value=Config.hps.speakers[1])
                    vc_input3 = gr.Audio(label="Input Audio")
                    vc_submit = gr.Button("Convert", variant="primary")
                    vc_output1 = gr.Textbox(label="Output Message")
                    vc_output2 = gr.Audio(label="Output Audio")

        tts_submit.click(Config.tts_fn, [tts_input1, tts_input2, tts_input3], [
                         tts_output1, tts_output2]

                         )
        vc_submit.click(Config.vc_fn, [vc_input1, vc_input2, vc_input3], [
            vc_output1, vc_output2]
        )

    app.queue(concurrency_count=2)
    gr.close_all()
    app.launch(server_name='0.0.0.0',  show_api=False)
