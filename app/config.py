import os

from loguru import logger
from app.util import get_hparams_from_file, get_paths
import re
from re import Pattern
import onnxruntime as ort


class Config:
    hps: dict = None
    pattern: Pattern = None
    # symbol_to_id:dict = None
    speaker_choices: list = None
    ort_sess: ort.InferenceSession = None

    @classmethod
    def init(cls):

        logger.add(
            "vits_infer.log",  rotation="10 MB", encoding="utf-8", enqueue=True, retention="30 days"
        )

        model_path, cfg_path = get_paths()



        assert os.path.exists(cfg_path), "config file not found"
        cls.hps = get_hparams_from_file(cfg_path)
        brackets = ['（', '[', '『', '「', '【', ")", "】", "]", "』", "」", "）"]
        cls.pattern = re.compile('|'.join(map(re.escape, brackets)))
        cls.speaker_choices = list(
            map(lambda x: str(x[0])+":"+x[1], enumerate(cls.hps.speakers)))
        
        cls.setup_model(model_path)

        logger.info(
            f"Config init, model_path: {model_path}, cfg_path: {cfg_path}, speaker_choices: {cls.speaker_choices}"
        )
        logger.info("Config init done")

    @classmethod
    def setup_model(cls, model_path: str):
        cls.ort_sess = ort.InferenceSession(model_path)
