import sys
sys.path.append('..')



import gradio as gr
from text import text_to_sequence
from config import Config
from app.util import intersperse
from loguru import logger
from app.util import find_path_by_suffix, time_it
import pathlib
import numpy as np

def text_to_seq(text: str):
    text = Config.pattern.sub(' ', text).strip()
    text_norm = text_to_sequence(
        text, Config.hps.symbols,  Config.hps.data.text_cleaners)
    if Config.hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    return text_norm


@time_it
def tts_fn(text, speaker_id, speed=1.0):

    if len(text) > 500:
        return "Error: Text is too long, please down it to 200 characters", None

    seq = text_to_seq(text)
    x = np.array([seq], dtype=np.int64)
    x_len = np.array([x.shape[1]], dtype=np.int64)
    sid = np.array([speaker_id], dtype=np.int64)
    speed = 1/speed
    # logger.debug(
    #     f"speed {speed}"
    # )
    scales = np.array([0.667, 1.0, speed], dtype=np.float32)
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
                    # tts_input3 = gr.Slider(
                    #     label="Speed", value=1, minimum=0.2, maximum=3, step=0.1)

                    tts_submit = gr.Button("Generate", variant="primary")
                    tts_output1 = gr.Textbox(label="Output Message")
                    tts_output2 = gr.Audio(label="Output Audio")

                    inputs = [
                        tts_input1, tts_input2
                    ]
                    outputs = [
                        tts_output1, tts_output2]

        tts_submit.click(tts_fn, inputs=inputs, outputs=outputs)

    app.queue(concurrency_count=2)
    app.launch(server_name='0.0.0.0', show_api=False)


def get_paths() -> tuple[str, str]:
    dir_path = pathlib.Path(__file__).parent.absolute() / ".model"
    dir_path.mkdir(
        parents=True, exist_ok=True
    )

    model_path = find_path_by_suffix(dir_path, "onnx")
    config_path = find_path_by_suffix(dir_path, "json")
    if not model_path or not config_path:
        model_path = dir_path / "model.onnx"
        config_path = dir_path / "config.json"
        logger.warning(
            "unable to find model or config, try to download default model and config"
        )
        import requests
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            model_url = r"https://ccdesue-my.sharepoint.com/personal/ccds_ccdesue_onmicrosoft_com/_layouts/15/download.aspx?share=ET6NYdpJa7tBsM3vICCcqbsBpnEdRnnTq9m9P26ENutFVA"
            config_url = r"https://ccdesue-my.sharepoint.com/personal/ccds_ccdesue_onmicrosoft_com/_layouts/15/download.aspx?share=EdsTRVK3l-FJnKNt6fJ15-8BIvq_dsqMIJuAz29oEezqTg"
            executor.submit(requests.get, model_url, stream=True).add_done_callback(
                lambda future: open(str(model_path), 'wb').write(future.result().content))
            executor.submit(requests.get, config_url, stream=True).add_done_callback(
                lambda future: open(str(config_path), 'wb').write(future.result().content))

            # wait for t1
            # t1.result()
            # # wait for t2
            # t2.result()

    model_path = str(model_path)
    config_path = str(config_path)
    logger.info(f"model path: {model_path} config path: {config_path}")
    return model_path, config_path


def main():

    model_path, config_path = get_paths()

    Config.init(model_path=model_path, cfg_path=config_path)
    set_gradio_view()


if __name__ == '__main__':
    main()
