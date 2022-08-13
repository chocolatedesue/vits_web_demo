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
    text_norm = text_to_sequence(text,  hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def tts_fn(text, speaker_id):
    if len(text) > 150:
        return "Error: Text is too long", None
    stn_tst = get_text(text)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()
    return "Success", (hps.data.sampling_rate, audio)


def vc_fn(original_speaker_id, target_speaker_id, input_audio):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    duration = audio.shape[0] / sampling_rate
    if duration > 30:
        return "Error: Audio is too long", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != hps.data.sampling_rate:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
    y = torch.FloatTensor(audio)
    y = y.unsqueeze(0)
    spec = spectrogram_torch(y, hps.data.filter_length,
                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                             center=False)
    spec_lengths = LongTensor([spec.size(-1)])
    sid_src = LongTensor([original_speaker_id])
    sid_tgt = LongTensor([target_speaker_id])
    with no_grad():
        audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
            0, 0].data.cpu().float().numpy()
    return "Success", (hps.data.sampling_rate, audio)

def parse_args(parser):
    parser.add_argument("--config",'-c',default="~/.model/model.pth")
    parser.add_argument("--model",'-m',default="~/.model/config.json")
    return parser


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="define the config_path and model_path")
    parser = parse_args(parser)
    
    args = parser.parse_args()

    config_path = args.config
    model_path = args.model

    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    utils.load_checkpoint(model_path, model, None)
    model.eval()

    app = gr.Blocks()

    with app:
        gr.Markdown("a demo of web service of vits, thanks to @CjangCjengh, copy from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts)")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Column():
                    tts_input1 = gr.TextArea(label="Text (150 words limitation)", value="こんにちは、あやち寧々です。")
                    tts_input2 = gr.Dropdown(label="Speaker", choices=hps.speakers, type="index", value=hps.speakers[0])
                    tts_submit = gr.Button("Generate", variant="primary")
                    tts_output1 = gr.Textbox(label="Output Message")
                    tts_output2 = gr.Audio(label="Output Audio")
            with gr.TabItem("Voice Conversion"):
                gr.Markdown("To enable this, please install ffmpege in the server")
                with gr.Column():
                    vc_input1 = gr.Dropdown(label="Original Speaker", choices=hps.speakers, type="index",
                                            value=hps.speakers[0])
                    vc_input2 = gr.Dropdown(label="Target Speaker", choices=hps.speakers, type="index",
                                            value=hps.speakers[1])
                    vc_input3 = gr.Audio(label="Input Audio (30s limitation)")
                    vc_submit = gr.Button("Convert", variant="primary")
                    vc_output1 = gr.Textbox(label="Output Message")
                    vc_output2 = gr.Audio(label="Output Audio")

        tts_submit.click(tts_fn, [tts_input1, tts_input2], [tts_output1, tts_output2])
        vc_submit.click(vc_fn, [vc_input1, vc_input2, vc_input3], [vc_output1, vc_output2])

    app.launch(server_name='0.0.0.0')
