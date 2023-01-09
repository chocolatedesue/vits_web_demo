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
from mel_processing import spectrogram_torch
# import argparse
import re

brackets = ['（', '[', '『', '「', '【', ")", "】", "]", "』", "」", "）"]
pattern = re.compile('|'.join(map(re.escape, brackets)))


def time_it(func: callable):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"Time cost: {end-start}")
        return res
    return wrapper


def text_cleanner(text: str):
    # text = re.sub(pattern, ' ', text)
    text = pattern.sub(' ', text)
    return text.strip()


def get_text(text):
    global symbols
    text_norm = text_to_sequence(
        text, symbols,  hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


@time_it
def tts_fn(text, speaker_id, speed=1.0):
    logger.debug(f"Text: {text}, Speaker ID: {speaker_id}, Speed: {speed}")
    if len(text) > 200:
        return "Error: Text is too long, please down it to 200 characters", None
    text = text_cleanner(text)
    stn_tst = get_text(text)
    with no_grad():
        global device
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0/speed)[0][
            0, 0].data.cpu().float().numpy()
    return "Success", (hps.data.sampling_rate, audio)


@logger.catch
@utils.time_it
def vc_fn(original_speaker_id, target_speaker_id, input_audio):
    logger.debug(
        f"Original Speaker ID: {original_speaker_id}, Target Speaker ID: {target_speaker_id}")
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    duration = audio.shape[0] / sampling_rate
    if duration > 3600:
        return "Error: Audio is too long, please down it to 3600s", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != hps.data.sampling_rate:
        audio = librosa.resample(
            audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
    # 将所有的向量送到cuda中
    global device
    y = torch.FloatTensor(audio).to(device)
    y = y.unsqueeze(0)
    spec = spectrogram_torch(y, hps.data.filter_length,
                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                             center=False).to(device)

    spec_lengths = LongTensor([spec.size(-1)]).to(device)
    sid_src = LongTensor([original_speaker_id]).to(device)
    sid_tgt = LongTensor([target_speaker_id]).to(device)
    with no_grad():
        audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
            0, 0].data.cpu().float().numpy()
    return "Success", (hps.data.sampling_rate, audio)


def setup_model():

    logger.warning("search default config and model")
    dir_path = pathlib.Path(__file__).parent.absolute() / ".model"
    dir_path.mkdir(
        parents=True, exist_ok=True
    )
    model_path = utils.find_path_by_suffix(dir_path, "pth")
    config_path = utils.find_path_by_suffix(dir_path, "json")
    if not config_path or not model_path:
        logger.warning(
            "unable to find model or config, try to download default model and config"
        )
        model_path = dir_path / "model.pth"
        config_path = dir_path / "config.json"
        utils.download_defaults(model_path=model_path, config_path=config_path)

    logger.debug(f"model_path: {model_path}, config_path: {config_path}")
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global hps  # 读取配置文件
    hps = utils.get_hparams_from_file(str(config_path))
    # global symbols
    symbols = hps.symbols
    global model  # 读取模型
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    utils.load_checkpoint(model_path, model, None)
    model.eval()


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
                        label="Speaker", choices=speaker_choices, type="index", value=speaker_choices[0])
                    tts_input3 = gr.Slider(
                        label="Speed", value=1, minimum=0.2, maximum=3, step=0.1)

                    tts_submit = gr.Button("Generate", variant="primary")
                    tts_output1 = gr.Textbox(label="Output Message")
                    tts_output2 = gr.Audio(label="Output Audio")

            with gr.TabItem("Voice Conversion"):
                gr.Markdown(
                    "***please upload wav file, other format is not supported now.***")
                with gr.Column():
                    vc_input1 = gr.Dropdown(label="Original Speaker", choices=hps.speakers, type="index",
                                            value=hps.speakers[0])
                    vc_input2 = gr.Dropdown(label="Target Speaker", choices=hps.speakers, type="index",
                                            value=hps.speakers[1])
                    vc_input3 = gr.Audio(label="Input Audio")
                    vc_submit = gr.Button("Convert", variant="primary")
                    vc_output1 = gr.Textbox(label="Output Message")
                    vc_output2 = gr.Audio(label="Output Audio")

        tts_submit.click(tts_fn, [tts_input1, tts_input2, tts_input3], [
                         tts_output1, tts_output2]

                         )
        vc_submit.click(vc_fn, [vc_input1, vc_input2, vc_input3], [
            vc_output1, vc_output2]
        )

    app.queue(concurrency_count=2)
    gr.close_all()
    app.launch(server_name='0.0.0.0', server_port=7860, show_api=False)
