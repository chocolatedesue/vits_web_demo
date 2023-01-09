from multiprocessing import Process
import gradio as gr
from text import text_to_sequence
from config import Config
from app.util import download_defaults, intersperse
from loguru import logger
from app.util import find_path_by_suffix, time_it
import numpy as np
import sys
sys.path.append('..')


def text_to_seq(text: str):
    text = Config.pattern.sub(' ', text).strip()
    text_norm = text_to_sequence(
        text, Config.hps.symbols,  Config.hps.data.text_cleaners)
    if Config.hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    return text_norm


@time_it
@logger.catch
def tts_fn(text, speaker_id, speed=1.0):

    if len(text) > 500:
        return "Error: Text is too long, please down it to 500 characters", None

    if Config.ort_sess is None:
        return "Error: model not loaded, please wait for a while or look the log", None

    seq = text_to_seq(text)
    x = np.array([seq], dtype=np.int64)
    x_len = np.array([x.shape[1]], dtype=np.int64)
    sid = np.array([speaker_id], dtype=np.int64)
    speed = 1/speed
    scales = np.array([0.667, speed, 0.8], dtype=np.float32)
    scales.resize(1, 3)
    ort_inputs = {
        'input': x,
        'input_lengths': x_len,
        'scales': scales,
        'sid': sid
    }
    audio = np.squeeze(Config.ort_sess.run(None, ort_inputs))
    audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
    audio = np.clip(audio, -32767.0, 32767.0)

    return "success", (Config.hps.data.sampling_rate, audio.astype(np.int16))


def set_gradio_view():
    app = gr.Blocks()

    with app:
        gr.Markdown(
            "a demo of web service of vits, thanks to @CjangCjengh, copy from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts)")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Column():
                    tts_input1 = gr.TextArea(
                        label="TTS_text", value="こんにちは、あやち寧々です。")
                    tts_input2 = gr.Dropdown(
                        label="Speaker", choices=Config.speaker_choices, type="index", value=Config.speaker_choices[0])
                    tts_input3 = gr.Slider(
                        label="Speed", value=1, minimum=0.2, maximum=3, step=0.1)

                    tts_submit = gr.Button("Generate", variant="primary")
                    tts_output1 = gr.Textbox(label="Output Message")
                    tts_output2 = gr.Audio(label="Output Audio")

                    inputs = [
                        tts_input1, tts_input2, tts_input3
                    ]
                    outputs = [
                        tts_output1, tts_output2]

        tts_submit.click(tts_fn, inputs=inputs, outputs=outputs)

    app.queue(concurrency_count=2)
    app.launch(server_name='0.0.0.0', show_api=True, share=False)


def main():
    # p = Process(target=Config.init)
    # p.start()
    Config.init()
    set_gradio_view()


if __name__ == '__main__':
    main()
