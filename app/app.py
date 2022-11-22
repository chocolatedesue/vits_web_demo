# import os

# os.system('cd monotonic_align && python setup.py build_ext --inplace && cd ..')

import librosa
import numpy as np
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
    # if len(text) > 150:
    #     return "Error: Text is too long", None
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
    parser.add_argument("--dir", '-d', default="/mydata")
    return parser


if __name__ == '__main__':
    import os

    def find_by_postfix(dir_path: str, postfix: str):
        for i in os.listdir(dir_path):
            res = i.split('.')[-1]
            if res == postfix:
                return os.path.join(dir_path, i)

        raise FileNotFoundError(
            f"Cann't find file endwith {postfix}, please check dir path")
    parser = argparse.ArgumentParser(
        description="define the config_path and model_path")
    parser = parse_args(parser)
    args = parser.parse_args()

    dir_path = args.dir
    config_path = find_by_postfix(dir_path, "json")
    model_path = find_by_postfix(dir_path, "pth")
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
    mychoices = []
    # for i in range(len(hps.speakers)):
    #     mychoices.ap
    for i, j in enumerate(hps.speakers):
        mychoices.append(str(i)+": "+j)

    with app:
        gr.Markdown(
            "a demo of web service of vits, thanks to @CjangCjengh, copy from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts)")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Column():
                    tts_input1 = gr.TextArea(
                        label="Text", value="こんにちは、あやち寧々です。")
                    tts_input2 = gr.Dropdown(
                        label="Speaker", choices=mychoices, type="index", value=mychoices[0])
                    tts_input3 = gr.Slider(
                        label="Speed", value=1, minimum=0.2, maximum=3, step=0.1)
                    # with gr.Accordion(label="Advanced Options", open=False):
                    #     temp_text_var = gr.Variable()
                    #     symbol_input = gr.Checkbox(
                    #         value=False, label="Symbol input")
                    #     symbol_list = gr.Dataset(label="Symbol list", components=[tts_input1],
                    #                              samples=[[x]
                    #                                       for x in symbols],
                    #                              elem_id=f"symbol-list{i}")
                    #     symbol_list_json = gr.Json(
                    #         value=symbols, visible=False)

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
                         tts_output1, tts_output2])
        vc_submit.click(vc_fn, [vc_input1, vc_input2, vc_input3], [
                        vc_output1, vc_output2])

    app.launch(server_name='0.0.0.0')
