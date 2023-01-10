import os
from pathlib import Path
from loguru import logger
import numpy as np
# from app import ,
# from app.util import get_hparams_from_file, get_paths, time_it
import requests
import torch
from tqdm.auto import tqdm
import re
import librosa
from re import Pattern
# import onnxruntime as ort
import threading
from mel_processing import spectrogram_torch
from text import text_to_sequence
import commons
from models import SynthesizerTrn
from utils import get_hparams_from_file
from utils import get_paths, time_it, load_checkpoint
MODEL_URL = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJlemNoNnJRRWRiM2NaZms/root/content'
CONFIG_URL = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhM1lzQ0Zzakl3dnQ5NDg/root/content'


class Config:
    hps: dict = None
    pattern: Pattern = None
    device: torch.device = None
    speaker_choices: list = None
    model: SynthesizerTrn = None
    model_is_ok: bool = False

    @classmethod
    def init(cls):

        brackets = ['（', '[', '『', '「', '【', ")", "】", "]", "』", "」", "）"]
        cls.pattern = re.compile('|'.join(map(re.escape, brackets)))
        cls.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        dir_path = Path(__file__).parent.absolute() / ".model"
        dir_path.mkdir(
            parents=True, exist_ok=True
        )
        model_path, config_path = get_paths(dir_path)

        if not model_path or not config_path:
            model_path = dir_path / "model.pth"
            config_path = dir_path / "config.json"
            logger.warning(
                "unable to find model or config, try to download default model and config"
            )
            cfg = requests.get(CONFIG_URL,  timeout=5).content
            with open(str(config_path), 'wb') as f:
                f.write(cfg)
            cls.setup_config(str(config_path))
            t = threading.Thread(target=cls.pdownload,
                                 args=(MODEL_URL, str(model_path)))
            t.start()

        else:
            cls.setup_config(str(config_path))
            cls.setup_model(str(model_path))

        # cls.speaker_choices = list(
        #     map(lambda x: str(x[0])+":"+x[1], enumerate(cls.hps.speakers)))

    @classmethod
    @time_it
    @logger.catch
    def setup_model(cls, model_path: str):
        cls.model = SynthesizerTrn(
            len(cls.hps.symbols),
            cls.hps.data.filter_length // 2 + 1,
            cls.hps.train.segment_size // cls.hps.data.hop_length,
            n_speakers=cls.hps.data.n_speakers,
            **cls.hps.model).to(cls.device)
        load_checkpoint(model_path, cls.model, None)
        cls.model.eval()

        cls.tts_fn("こにちわ", 0)

        sr = 22050

        # Frequency of the sine wave
        freq = 440

        # Duration of the sine wave (in seconds)
        duration = 1

        # Generate time samples for the sine wave
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Generate samples for the sine wave
        samples = np.sin(2 * np.pi * freq * t)

        cls.vc_fn(
            0, 1, samples
        )

        cls.model_is_ok = True

        logger.info(
            f"model init done with model path {model_path}"
        )

    @classmethod
    def setup_config(cls, config_path: str):
        cls.hps = get_hparams_from_file(config_path)
        cls.speaker_choices = list(
            map(lambda x: str(x[0])+":"+x[1], enumerate(cls.hps.speakers)))

        logger.info(
            f"config init done with config path {config_path}"
        )

    @classmethod
    def pdownload(cls, url: str, save_path: str, chunk_size: int = 8192):
        # copy from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_requests.py
        file_size = int(requests.head(url).headers["Content-Length"])
        response = requests.get(url, stream=True)
        with tqdm(total=file_size,  unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc="model download") as pbar:

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(chunk_size)
        cls.setup_model(save_path)

    @classmethod
    def get_text(cls, text):
        Config.pattern.sub(' ', text)
        text_norm = text_to_sequence(
            text, Config.hps.symbols,  Config.hps.data.text_cleaners)
        if Config.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        # text_norm = LongTensor(text_norm)
        return text_norm

    @classmethod
    @time_it
    @logger.catch
    def tts_fn(cls, text, speaker_id, speed=1.0):
        if not Config.model_is_ok:
            return "Error: Model is not loaded, please wait for a while and look the log", None

        logger.debug(f"Text: {text}, Speaker ID: {speaker_id}, Speed: {speed}")
        if len(text) > 200:
            return "Error: Text is too long, please down it to 200 characters", None

        with torch.no_grad():
            stn_tst = torch.LongTensor(cls.get_text(text)).to(Config.device)
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor(
                [stn_tst.size(0)]).to(Config.device)
            sid = torch.LongTensor([speaker_id]).to(Config.device)
            audio = Config.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0/speed)[0][
                0, 0].data.cpu().float().numpy()
        return "Success", (Config.hps.data.sampling_rate, audio)

    @classmethod
    @logger.catch
    @time_it
    def vc_fn(cls, original_speaker_id, target_speaker_id, input_audio):
        if not Config.model_is_ok:
            return "Error: Model is not loaded, please wait for a while and look the log", None

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
        if sampling_rate != Config.hps.data.sampling_rate:
            audio = librosa.resample(
                audio, orig_sr=sampling_rate, target_sr=Config.hps.data.sampling_rate)

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(Config.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, Config.hps.data.filter_length,
                                     Config.hps.data.sampling_rate, Config.hps.data.hop_length, Config.hps.data.win_length,
                                     center=False).to(Config.device)

            spec_lengths = torch.LongTensor([spec.size(-1)]).to(Config.device)
            sid_src = torch.LongTensor([original_speaker_id]).to(Config.device)
            sid_tgt = torch.LongTensor([target_speaker_id]).to(Config.device)
            audio = Config.model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        return "Success", (Config.hps.data.sampling_rate, audio)
