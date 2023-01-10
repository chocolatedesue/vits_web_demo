import os
from pathlib import Path
from loguru import logger
# from app import ,
# from app.util import get_hparams_from_file, get_paths, time_it
import requests
import torch
from tqdm.auto import tqdm
import re
from re import Pattern
import onnxruntime as ort
import threading
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

    @classmethod
    def init(cls):
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

        brackets = ['（', '[', '『', '「', '【', ")", "】", "]", "』", "」", "）"]
        cls.pattern = re.compile('|'.join(map(re.escape, brackets)))
        cls.speaker_choices = list(map(lambda x: str(x[0])+":"+x[1], enumerate(cls.hps.speakers)))

    @classmethod
    def setup_model(cls, model_path: str):
        cls.model = SynthesizerTrn(
            len(cls.hps.symbols),
            cls.hps.data.filter_length // 2 + 1,
            cls.hps.train.segment_size // cls.hps.data.hop_length,
            n_speakers=cls.hps.data.n_speakers,
            **cls.hps.model).to(cls.device)
        load_checkpoint(model_path, cls.model, None)
        cls.model.eval()

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
    @time_it
    @logger.catch
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
