from pathlib import Path

import numpy as np
import onnxruntime as ort
from loguru import logger
from pydub import AudioSegment

from .util import HParams, find_path_by_suffix, model_warm_up, get_hparams_from_file
from .util import time_it

MODEL_URL = r"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJmckZWcGdCR0xxLWJmU28/root/content"
CONFIG_URL = r"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhNEJ3enhhUHpqNE5EZWc/root/content"


class Config:
    model_dir_path: Path = None
    last_save_dir: Path = Path(__file__).parent.parent / "output"
    ort_sess: ort.InferenceSession = None
    model_is_ok: bool = False
    hps: HParams = None
    speaker_choices: list = ["none"]
    audio_data: np.array = None
    seg: AudioSegment = None

    @classmethod
    def init(cls, dir_path: Path):
        if not dir_path.is_dir():
            return "dir not exist"
        model_path = (find_path_by_suffix(dir_path, "onnx"))
        config_path = (find_path_by_suffix(dir_path, "json"))
        if not model_path or not config_path:
            return "unable to find model or config, with model_dir_path: {}".format(dir_path)
        try:
            cls.setup_config(str(config_path))
            cls.setup_model(str(model_path))
        except Exception as e:
            logger.error(e)
            return f"""maybe model is damaged
please remove the model and redownload again.
error: {str(e)}
            """
        return None

    @classmethod
    @time_it
    # @logger.catch
    def setup_model(cls, model_path: str):
        provider = ['DmlExecutionProvider', 'CPUExecutionProvider']
        so = ort.SessionOptions()
        # For CPUExecutionProvider you can change it to True
        so.enable_mem_pattern = False
        try:
            cls.ort_sess = ort.InferenceSession(
                model_path, providers=provider, sess_options=so)
            model_warm_up(cls.ort_sess, cls.hps)

            cls.model_is_ok = True
            # model_warm_up(cls.ort_sess, cls.hps)
        except Exception as e:
            logger.error(e)
            raise e

    @classmethod
    # @logger.catch
    def setup_config(cls, config_path: str):
        cls.hps = get_hparams_from_file(config_path)
        cls.speaker_choices = list(
            map(lambda x: str(x[0]) + ":" + x[1], enumerate(cls.hps.speakers)))

        logger.info(
            f"config init done with config path {config_path}"
        )

    # def save(self):
