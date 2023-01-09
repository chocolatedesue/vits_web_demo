import os

from loguru import logger 
from app.util import get_hparams_from_file
import re
from re import Pattern
import onnxruntime as ort


class Config:
    hps:dict = None
    pattern:Pattern = None
    symbol_to_id:dict = None
    speaker_choices:list = None 
    ort_sess:ort.InferenceSession = None
    
    @classmethod
    def init(cls,cfg_path:str,model_path:str):
        assert os.path.exists(cfg_path), "config file not found"
        cls.hps = get_hparams_from_file(cfg_path)
        symbols = cls.hps.symbols
        
        cls.symbol_to_id = {s: i for i, s in enumerate(symbols)}

        brackets = ['（', '[', '『', '「', '【', ")", "】", "]", "』", "」", "）"]
        cls.pattern = re.compile('|'.join(map(re.escape, brackets)))
        cls.speaker_choices = list(
        map(lambda x: str(x[0])+":"+x[1], enumerate(cls.hps.speakers)))

        logger.debug(
             f"Config init success, model_path: {model_path}, cfg_path: {cfg_path}, speaker_choices: {cls.speaker_choices}"
        )
        logger.debug(f"Config init success, hps: {cls.hps}")
        logger.debug(f"Config init success, symbol_to_id: {cls.symbol_to_id}")

        cls.setup_model(model_path)


    @classmethod
    def setup_model(cls,model_path:str):
        cls.ort_sess = ort.InferenceSession(model_path)

        
