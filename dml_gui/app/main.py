import pathlib
# import  time
import time
import webbrowser
from pathlib import Path
from tkinter import filedialog
# import chardet
import PySimpleGUI as sg
import numpy as np
import requests
from loguru import logger
from pydub import AudioSegment
from pydub.playback import play

from text import text_to_seq
# from config import /
from utils.config import MODEL_URL, CONFIG_URL, Config
from utils.util import find_path_by_suffix
from utils.util import ort_infer

# sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
# logger.remove()
logger.add(
    "log.log", rotation="1 MB", retention=2, enqueue=True, encoding="utf-8", level="INFO"
)

cur_dir = Path(__file__).parent.absolute()
 

def setup_defalut(window: sg.Window, dir_path: Path, chunk_size=8192):
    # window["status"].Update("downloading defaults",background_color="yellow", visible=True)
    window["model_status"].Update("downloading", visible=True)
    try:
        cfg = requests.get(CONFIG_URL, timeout=5).content
        with open(str(dir_path / "config.json"), 'wb') as f:
            f.write(cfg)
        Config.setup_config(str(dir_path / "config.json"))
        window["speaker"].update(
            value=Config.speaker_choices[0], values=Config.speaker_choices)
        model = requests.get(MODEL_URL, stream=True)
        total_length = int(model.headers.get('content-length'))
        window["download_progress"].update(0, total_length, visible=True)

        with open(str(dir_path / "model.onnx"), 'wb') as f:
            tot = 0
            for chunk in model.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    tot += chunk_size
                    window["download_progress"].update(
                        tot)
                    # window.refresh()
        window['dir_info'].update(str(dir_path))
        window["model_status"].update("finish downloaded", visible=True, background_color="green")
    except Exception as e:
        logger.error(e)
        window["model_status"].update(visible=False)
        window["df_model_status"].update("downaload fail",visible=True, background_color="red" )
        text =     f"""download failed, please check your network
error: {e}
            """
        lay = [sg.Multiline(text, auto_size_text=True, size=(50, 15), no_scrollbar=True,background_color=sg.theme_background_color(),border_width=0 ,text_color="white",disabled=True)]
        sg.Window('download fail', [lay],finalize=True).read(close=True)


        
        return 

    Config.model_dir_path = dir_path
    window.start_thread(lambda: window.write_event_value("-init_msg-", Config.init(Config.model_dir_path)),
                        ('-init-', '-init_end-'))


def tts_fn(window: sg.Window, speed: float):
    window["status"].update("doing inference", visible=True, background_color="yellow",text_color="black")
    if not Config.model_is_ok:
        window["status"].update("model not loaded", visible=True, background_color="red")
        return
    text = window["input"].get()
    if not text:
        window["status"].update("input is empty", visible=True, background_color="red")
        return

    seq = text_to_seq(text, Config.hps)

    x = np.array([seq], dtype=np.int64)
    x_len = np.array([x.shape[1]], dtype=np.int64)
    sid = np.array([Config.speaker_choices.index(
        window["speaker"].get())], dtype=np.int64)
    # speed = 1 / speed
    # speed = int(window["speed"].read())
    scales = np.array([0.667, speed, 0.8], dtype=np.float32)
    scales.resize(1, 3)
    ort_inputs = {
        'input': x,
        'input_lengths': x_len,
        'scales': scales,
        'sid': sid
    }
    start_time = time.perf_counter()
    audio_data = ort_infer(Config.ort_sess, ort_inputs)
    end_time = time.perf_counter()
    Config.seg = AudioSegment(
        audio_data.tobytes(), sample_width=2, frame_rate=Config.hps.data.sampling_rate, channels=1)
    # Config.seg.export("tmp.wav", format="wav")
    global cur_dir
    Config.seg.export(str(cur_dir / "tmp.wav"), format="wav")
    window["status"].update(f"generate {Config.seg.duration_seconds}, elapse: {end_time - start_time} ",
                            visible=True,
                            background_color="green")


def update_model_status(window: sg.Window):
    window["speaker"].update(
        value=Config.speaker_choices[0], values=Config.speaker_choices, visible=True)
    window["df_model_status"].Update("model loaded", visible=True, background_color="green")


# Create a layout for the GUI
layout = [
    [sg.Text('vits_onnx with DirectMl', font='Any 20'), sg.ReadButton('Help', key='Help')],
    [sg.Text('Choose the dir of model and config (search by suffix: [.onnx|.json]):')],
    [sg.Input(key="dir_info", enable_events=True), sg.FolderBrowse(
        key="select_dir")],
    [sg.Button(button_text="load", key="load",tooltip="can also be used to manually clear infer memory usage"), sg.Button(button_text="load default model from Internet",tooltip="if download is slow, you download it manually or restart exe to download"),
     sg.Text("default model has ~110M", key="desc_model"),
     sg.Text("", visible=False, key="df_model_status")],
    [sg.ProgressBar(key="download_progress", orientation="h", size=(20, 20), visible=False, max_value=100,
                    bar_color=('green', 'white')), sg.Text("status: ", key="model_status", visible=False)],
    [sg.Text("speaker"), sg.Combo(Config.speaker_choices,
                                  key="speaker", default_value=Config.speaker_choices[0], size=(20, 1))],
    [sg.Multiline("text_input_area, attention: inputs vary according model config, like  [JA]こんにちは。[JA][ZH]你好。[ZH]",no_scrollbar=True,disabled=True,background_color=sg.theme_background_color(),text_color="white",size=(50, 3),  auto_size_text=True,border_width=0)],
    [sg.Multiline(key="input", size=(50, 3), autoscroll=True,
                  auto_size_text=True, no_scrollbar=True, default_text="わたしの趣味はたくさんあります。でも、一番好きな事は写真をとることです。")],
    [sg.Text("speed rate:"),
     sg.Slider(range=(0.1, 4), default_value=1, key="speed", orientation="h", size=(20, 15), resolution=0.1)],
    # sg.Slider(key="speed", range=(0.1, 3), default_value=1, resolution=0.1, orientation="h", )],
    # sg.SaveAs(key="Save", button_text="Save", file_types=(("WAV", "*.wav")
    [sg.Submit(button_text="TTS"), sg.Submit(
        button_text="Play"), sg.Submit(button_text="Save", key="Save"),
     sg.Submit(button_text="Open last save folder", key="Open last save folder")],
    [sg.Text("status: model not loaded", key="status", visible=False, background_color="green")],
]

# Create the window from the layout
window = sg.Window('onnx_directml_demo', layout=layout,finalize=True
                   )

# focus_element = window["dir_info"]
# focus_element.bind('<FocusOut>', '+FOCUS OUT')

Config.last_save_dir.mkdir(parents=True, exist_ok=True)

while True:
    event, values = window.read()
    # logger.debug(
    #     f"event: {event}, values: {values}"
    # )

    if event in (sg.WIN_CLOSED, 'Cancel'):
        break


    elif event == 'load default model from Internet':
        window["desc_model"].update(visible=False)
        window["df_model_status"].update("model is loading",visible=True,background_color="yellow", text_color="black")
        default_dir = pathlib.Path(__file__).parent / ".model"
        default_dir.mkdir(exist_ok=True, parents=True)
        window["dir_info"].update(str(default_dir))
        window.refresh()
        cfg_path = find_path_by_suffix(default_dir, "json")
        model_path = find_path_by_suffix(default_dir, "onnx")
        logger.warning(f"cfg_path: {cfg_path}, model_path: {model_path}")
        if not cfg_path or not model_path:
            window.start_thread(lambda: setup_defalut(
                window, default_dir), ('-THREAD-', '-THEAD ENDED-'))

        else:
            Config.model_dir_path = default_dir
            window.start_thread(lambda: window.write_event_value("-init_msg-", Config.init(Config.model_dir_path)),
                                ('-init-', '-init_end-'))


    elif event == 'TTS':
        # tts_fn(window)
        window.start_thread(lambda: tts_fn(window, 1 / values["speed"]), ('--', '---'))

    elif event == 'Play':
        if Config.seg is not None:
            # global cur_dir
            path = str(cur_dir / "tmp.wav")
            try:
                webbrowser.open(
                    f"file://{path}", new=0, autoraise=True)
            except Exception as e:
                logger.error(e)
                window.start_thread(lambda: play(Config.seg), ('--', '---'))
        else:
            window["status"].update("audio data is None, please infer first !", background_color="red",
                                    visible=True)

    elif event == "Save":
        # f = filedialog.asksaveasfile(mode='w', defaultextension=".wav")
        if not Config.seg:
            sg.popup("please infer first !")
            continue 
        f_path = filedialog.asksaveasfilename(defaultextension=".wav", initialfile="output.wav",
                                              initialdir=str(Config.last_save_dir))
        if f_path:
            Config.last_save_dir = Path(f_path).parent
            Config.seg.export(f_path, format="wav")

    elif event == "load":
        window["desc_model"].update(visible=False)
        window["df_model_status"].update("model is loading",visible=True,background_color="yellow", text_color="black",)
        window.refresh()
        Config.model_dir_path = Path(values['dir_info'])
        window.start_thread(lambda: window.write_event_value("-init_msg-", Config.init(Config.model_dir_path)),
                            ('-init-', '-init_end-'))
        # window.refresh()

    elif "-init_msg-" in event:
        # try:
        msg = values["-init_msg-"]
        if msg:
            sg.popup(msg,keep_on_top=True)
            window["df_model_status"].update("model load failed",visible=True,background_color="red", text_color="black" )
        else:
            update_model_status(window)
            window["dir_info"].update(str(Config.model_dir_path), visible=True)


    elif event == "Open last save folder":
        if Config.last_save_dir:
            webbrowser.open(str(Config.last_save_dir), new=0, autoraise=True)
        else:
            sg.popup("no last save dir")

    # elif event == "-tts_infer-":

    elif event == "Help":
        text = """
        1. This is only used for study and research, please do not use it for commercial purposes.
        2. github repo:  https://github.com/chocolatedesue/vits_web_demo 
        3. gitee repo: https://gitee.com/ccdesue/vits_web_demo
        4. If you have any questions, please refer readme.md 
        5. if bug, please see the log.log file at the exe dir
    """
        # sg.popup(text,title="Help", keep_on_top=True, modal=True)
        # choice, _ = sg.Window('Help', [sg.InputText('You can select this text', use_readonly_for_disable=True, disabled=True, key='-IN-')])

        lay = [sg.Multiline(text, auto_size_text=True, size=(50, 15), no_scrollbar=True,background_color=sg.theme_background_color(),border_width=0 ,text_color="white",disabled=True)]
        sg.Window('Help', [lay],finalize=True).read(close=True)

# with open ("infer_config.json","w", encoding="utf-8") as f:


# Close the window
window.close()
