import time

from loguru import logger
from text import text_to_seq
from config import Config
import PySimpleGUI as sg
import pathlib
import requests
# import  time
import sys
from config import MODEL_URL, CONFIG_URL

sys.path.append(pathlib.Path(__file__).parent.parent.absolute().as_posix())
from pathlib import Path

logger.add(
    "log.log", rotation="1 MB", retention="10 days", enqueue=True, encoding="utf-8", level="INFO"
)


def setup_defalut(window: sg.Window, dir_path: Path):
    window["status"].Update("downloading defaults")
    window["download_progress"].update(0, visible=True)
    window["model_status"].Update("downloading", visible=True)

    try:
        cfg = requests.get(CONFIG_URL, timeout=5).content
        with open(str(dir_path / "config.json"), 'wb') as f:
            f.write(cfg)
        Config.setup_config(str(dir_path / "config.json"))
        window["speaker"].update(value=Config.speaker_choices[0], values=Config.speaker_choices)
    except Exception as e:
        logger.error(e)
        window["status"].Update("download failed")

    time.sleep(10)
    window["download_progress"].update(50)

    window["model_status"].Update("done")


def tts_fn(window: sg.Window):
    window["status"].update("doing inference", visible=True)
    if not Config.model_is_ok:
        window["status"].update("model not loaded")
        return
    import multiprocessing
    text = window["input"].get()
    if not text:
        window["status"].update("input is empty")
        return
    import numpy as np
    seq = text_to_seq(text, Config.hps)

    x = np.array([seq], dtype=np.int64)
    x_len = np.array([x.shape[1]], dtype=np.int64)
    sid = np.array([Config.speaker_choices.index(window["speaker"])], dtype=np.int64)
    # speed = 1 / speed
    speed = int(window["speed"].get())
    scales = np.array([0.667, speed, 0.8], dtype=np.float32)
    scales.resize(1, 3)
    ort_inputs = {
        'input': x,
        'input_lengths': x_len,
        'scales': scales,
        'sid': sid
    }
    window["status"].update("done", visible=True)


# sg.popup('TTS')


def update_model_status(window: sg.Window):
    window["speaker"].update(value=Config.speaker_choices[0], values=Config.speaker_choices)
    window["model_status"].Update("model loaded")


# Create a layout for the GUI
layout = [
    [sg.Text('vits_onnx with DirectMl', font='Any 20')],
    [sg.Text('Choose the dir of model and config (search by suffix: [.onnx|.json]):')],
    [sg.Input(key="dir_info", enable_events=True), sg.FolderBrowse(
        key="select_dir")],
    [sg.Button(button_text="load default model from Internet"), sg.Text("")],
    [sg.ProgressBar(key="download_progress", orientation="h", size=(20, 20), visible=False, max_value=100,
                    bar_color=('green', 'white')), sg.Text("status: ", key="model_status", visible=False)],
    [sg.Text("speaker"), sg.Combo(Config.speaker_choices,
                                  key="speaker", default_value=Config.speaker_choices[0], size=(20, 1))],
    [sg.Text("text_input_area")],
    [sg.Multiline(key="input", size=(50, 3), autoscroll=True,
                  auto_size_text=True, no_scrollbar=True)],
    [sg.Text("speaking speed"),
     sg.Slider(key="speed", range=(0.1, 3), default_value=1, resolution=0.1, orientation="h", )],
    [sg.Submit(button_text="TTS"), sg.Submit(button_text="Play")],
    [sg.Text("status: model not loaded", key="status", visible=False)],

]

# Create the window from the layout
window = sg.Window('Demo', layout=layout, ttk_theme="clam")

# Run the event loop to process user input
while True:
    event, values = window.read()
    logger.debug(
        f"event: {event}, values: {values}"
    )

    if event in (sg.WIN_CLOSED, 'Cancel'):
        break
    elif event == 'dir_info':
        dir_path = Path(values['dir_info'])
        msg = Config.init(dir_path)
        if msg:
            sg.popup(msg)
        else:
            update_model_status(window)

    elif event == 'load default model from Internet':

        default_dir = pathlib.Path(__file__).parent / ".model"
        window['dir_info'].update(str(default_dir))
        default_dir.mkdir(exist_ok=True, parents=True)
        window.start_thread(lambda: setup_defalut(window, default_dir), ('-THREAD-', '-THEAD ENDED-'))

    elif event == 'TTS':
        # tts_fn(window)
        window.start_thread(lambda: tts_fn(window), ('--', '---'))

    elif event == 'Play':
        sg.popup('Play')

# Close the window
window.close()
