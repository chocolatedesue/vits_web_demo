# import os

# os.system('cd monotonic_align && python setup.py build_ext --inplace && cd ..')

from typing import Optional
import librosa
import pathlib
from loguru import logger
import numpy as np
import requests
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
from models import SynthesizerTrn
from text import text_to_sequence
from mel_processing import spectrogram_torch
import argparse


def get_text(text):
    global symbols
    text_norm = text_to_sequence(
        text, symbols,  hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def tts_fn(text, speaker_id, speed=1.0):

    assert isinstance(text, str), f"text must be str, but got {type(text)}"
    assert isinstance(
        speaker_id, int), f"speaker_id must be int, but got {type(speaker_id)}"
    # assert isinstance(speed, float), f"speed must be float, but got {type(speed)}"
    if isinstance(speed, str):
        speed = float(speed)
    stn_tst = get_text(text)
    with no_grad():
        global device
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0/speed)[0][
            0, 0].data.cpu().float().numpy()
    return "Success", (hps.data.sampling_rate, audio)


def vc_fn(original_speaker_id, target_speaker_id, input_audio):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    # duration = audio.shape[0] / sampling_rate
    # if duration > 30:
    #     return "Error: Audio is too long", None
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


def parse_args(parser):
    # parser.add_argument("--config",'-c',default="~/.model/model.pth")
    # parser.add_argument("--model",'-m',default="~/.model/config.json")

    parser.add_argument("--dir", '-d',  required=False, default=None, type=str)
    return parser


def find_by_postfix(dir_path: Optional[str], postfix: Optional[str]):
    # assert os.path.exists(dir_path), f"{dir_path} not exists"
    if not dir_path or not postfix:
        return None
    if not os.path.exists(dir_path):
        return None
    assert isinstance(
        dir_path, str), f"dir_path must be str, but got {type(dir_path)}"
    assert isinstance(
        postfix, str), f"postfix must be str, but got {type(postfix)}"
    for i in os.listdir(dir_path):
        res = i.split('.')[-1]
        if res == postfix:
            return os.path.join(dir_path, i)
    # else:
    return None
    # raise FileNotFoundError(
    #     f"Cann't find file endwith {postfix}, please check dir path")


def save_model_and_config(model_bytes, config_bytes):
    model_dir = pathlib.Path.cwd() / ".model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model.pth"
    config_path = model_dir / "config.json"
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    with open(config_path, "wb") as f:
        f.write(config_bytes)
    return str(model_path), str(config_path)


if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser(
        description="define the config_path and model_path")
    parser = parse_args(parser)
    args = parser.parse_args()
    dir_path = args.dir
    config_path = find_by_postfix(dir_path, "json")
    model_path = find_by_postfix(dir_path, "pth")

    if not config_path or not model_path:
        logger.warning("use default config and model")
        model_dir = pathlib.Path.cwd() / ".model"
        config_path = find_by_postfix(str(model_dir), "json")
        model_path = find_by_postfix(str(model_dir), "pth")
        if not config_path or not model_path:
            model_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJiTzdqanlEQXNyWDV4bDA/root/content'
            config_url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhNEJ3enhhUHpqNE5EZWc/root/content'
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                model = executor.submit(requests.get, model_url)
                config = executor.submit(requests.get, config_url)
                model = model.result().content
                config = config.result().content
                model_path, config_path = save_model_and_config(model, config)
        # else:
    logger.info(f"model_path: {model_path}, config_path: {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hps = utils.get_hparams_from_file(config_path)
    global symbols
    symbols = hps.symbols
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    utils.load_checkpoint(model_path, model, None)
    model.eval()

    app = gr.Blocks()
    # mychoice = map(lambda x : +x)
    speaker_choices = []
    # for i in range(len(hps.speakers)):
    #     mychoices.ap
    # for i, j in enumerate(hps.speakers):
    #     speaker_choices.append(str(i)+":"+j)

    speaker_choices = list(
        map(lambda x: str(x[0])+":"+x[1], enumerate(hps.speakers)))

    with app:
        gr.Markdown(
            "a demo of web service of vits, thanks to @CjangCjengh, copy from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts)")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Column():
                    tts_input1 = gr.TextArea(
                        label="Text", value="こんにちは、あやち寧々です。")
                    tts_input2 = gr.Dropdown(
                        label="Speaker", choices=speaker_choices, type="index", value=speaker_choices[0])
                    tts_input3 = gr.Slider(
                        label="Speed", value=1, minimum=0.2, maximum=3, step=0.1)

                    tts_submit = gr.Button("Generate", variant="primary")
                    tts_output1 = gr.Textbox(label="Output Message")
                    tts_output2 = gr.Audio(label="Output Audio")

            with gr.TabItem("Voice Conversion"):
                gr.Markdown(
                    "To enable this, please install ffmpeg in the server")
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
                         tts_output1, tts_output2],
                         api_name="tts"

                         )
        vc_submit.click(vc_fn, [vc_input1, vc_input2, vc_input3], [
            vc_output1, vc_output2], api_name="vc"
        )

    app.queue(concurrency_count=2)
    app.launch(server_name='0.0.0.0')
