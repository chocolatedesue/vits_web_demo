# from multiprocessing import Process
import numpy as np
from .util import find_path_by_suffix, time_it
from loguru import logger
from .util import intersperse
from .config import Config
from .text import text_to_sequence
import gradio as gr
# import sys
# sys.path.append('..')


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

    if len(text) > 300:
        return "Error: Text is too long, please down it to 300 characters", None

    if not Config.model_is_ok:
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


def setup_elements():
    tts_input1 = gr.TextArea(
        label="TTS_text", value="わたしの趣味はたくさんあります。でも、一番好きな事は写真をとることです。")
    tts_input2 = gr.Dropdown(
        label="Speaker", choices=Config.speaker_choices, type="index", value=Config.speaker_choices[0])
    tts_input3 = gr.Slider(
        label="Speed", value=1, minimum=0.2, maximum=3, step=0.1)

    tts_output1 = gr.Textbox(label="Output Message")
    tts_output2 = gr.Audio(label="Output Audio")
    return [tts_input1, tts_input2, tts_input3], [tts_output1, tts_output2]


def set_infer_view():
    inputs, outputs = setup_elements()
    demo = gr.Interface(tts_fn, inputs, outputs,
                        # examples=[
                        #    [ "わたしの趣味はたくさんあります。でも、一番好きな事は写真をとることです。", 0, 1.0]
                        # ],
                        )
    # gr.Examples(
    #     examples=[
    #         ["わたしの趣味はたくさんあります。でも、一番好きな事は写真をとることです。", 0, 1.0]
    #     ],
    #     fn=tts_fn,
    #     inputs=inputs, outputs=outputs,
    #     cache_examples=True
    # )
    global args
    demo.launch(
        show_api=True,share=args.share
    )


def set_gradio_view():
    app = gr.Blocks()

    with app:
        gr.Markdown(
            "a demo of web service of vits, thanks to @CjangCjengh, copy from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts)")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Column():

                    inputs, outputs = setup_elements()
                    tts_submit = gr.Button("Generate", variant="primary")

        tts_submit.click(tts_fn, inputs=inputs, outputs=outputs)
    #     gr.Examples(
    #     examples=[
    #         ["わたしの趣味はたくさんあります。でも、一番好きな事は写真をとることです。", 0, 1.0]
    #     ],
    #     fn=tts_fn,
    #     inputs=inputs, outputs=outputs,
    #     cache_examples=True
    # )

    app.queue(concurrency_count=3)

    global args
    app.launch(server_name='0.0.0.0', show_api=False,
               share=args.share)


def main():

    Config.init()
    set_gradio_view()
    # set_infer_view()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', '-s', type=bool, default=False)
    args = parser.parse_args()
    main()
