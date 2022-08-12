# from distutils.command.config import config
import os

from models import SynthesizerTrn
from text.symbols import symbols
import gradio as gr
# from scipy.io.wavfile import write
import utils
import torch
from text import text_to_sequence
import commons
import model


def get_text(text):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def tts_fn(text, speaker_id):
    # pass
    stn_tst = get_text(text)
    # with torch.no_grad():
    #     x_tst = stn_tst.unsqueeze(0)
    #     x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    #     sid = torch.LongTensor([speaker_id])
    #     audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
    #         0, 0].data.cpu().float().numpy()
    # return hps.data.sampling_rate, audio
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([speaker_id])
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    return 22050, audio


def vc_fn(original_speaker_id, target_speaker_id, input_audio):
    pass
    # sampling_rate, audio = input_audio
    # y = torch.FloatTensor(audio.astype(np.float32)) / hps.data.max_wav_value
    # y = y.unsqueeze(0)

    # spec = spectrogram_torch(y, hps.data.filter_length,
    #                          hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
    #                          center=False)
    # spec_lengths = LongTensor([spec.size(-1)])
    # sid_src = LongTensor([original_speaker_id])
    # sid_tgt = LongTensor([target_speaker_id])
    # with no_grad():
    #     audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
    #         0, 0].data.cpu().float().numpy()
    # return hps.data.sampling_rate, audio


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "model/all_config.json"
    model_path = "model/all_G_35000.pth"   
    hps = utils.get_hparams_from_file(config_path)

    
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    
    utils.load_checkpoint(model_path, net_g, None)
    
    net_g.eval()

    

    app = gr.Blocks()

    with app:
        # gr.Markdown("# Moe Japanese TTS And Voice Conversion Using VITS Model\n\n"
        #             "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=skytnt.moegoe)\n\n"
        #             "unofficial demo for [https://github.com/CjangCjengh/MoeGoe](https://github.com/CjangCjengh/MoeGoe)"
        #             )
        gr.Markdown("a demo of my vits")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Column():
                    tts_input1 = gr.TextArea(label="Text", value="こんにちは")
                    tts_input2 = gr.Dropdown(label="Speaker", choices=hps.speakers, type="index", value=hps.speakers[0])
                    tts_submit = gr.Button("Generate", variant="primary")
                    tts_output = gr.Audio(label="Output Audio")
            with gr.TabItem("Voice Conversion"):
                with gr.Column():
                    vc_input1 = gr.Dropdown(label="Original Speaker", choices=hps.speakers, type="index",
                                            value=hps.speakers[0])
                    vc_input2 = gr.Dropdown(label="Target Speaker", choices=hps.speakers, type="index",
                                            value=hps.speakers[1])
                    vc_input3 = gr.Audio(label="Input Audio")
                    vc_submit = gr.Button("Convert", variant="primary")
                    vc_output = gr.Audio(label="Output Audio")

        tts_submit.click(tts_fn, [tts_input1, tts_input2], [tts_output])
        vc_submit.click(vc_fn, [vc_input1, vc_input2, vc_input3], [vc_output])

    app.launch()
