from pathlib import Path

from util import HParams, get_hparams_from_file, model_warm_up, time_it, find_path_by_suffix
from loguru import logger
import onnxruntime as ort

MODEL_URL = r"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJmckZWcGdCR0xxLWJmU28/root/content"
CONFIG_URL = r"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhNEJ3enhhUHpqNE5EZWc/root/content"


class Config:
    ort_sess: ort.InferenceSession = None
    model_is_ok: bool = False
    hps: HParams = None
    speaker_choices: list = ["none"]

    @classmethod
    def init(cls, dir_path: Path):
        if not dir_path.is_dir():
            return "dir not exist"

        model_path = find_path_by_suffix(dir_path, "onnx")
        config_path = find_path_by_suffix(dir_path, "json")
        if not model_path or config_path:
            return "unable to find model or config, with dir_path: {}".format(dir_path)
        try:
            cls.setup_config(config_path)
            cls.setup_model(model_path)
        except Exception as e:
            logger.error(e)
            return "unable to setup model or config, may be the model is damaged"
        return None

    @classmethod
    @time_it
    @logger.catch
    def setup_model(cls, model_path: str):
        cls.ort_sess = ort.InferenceSession(model_path)
        cls.model_is_ok = True
        model_warm_up(cls.ort_sess, cls.hps)

    @classmethod
    @logger.catch
    def setup_config(cls, config_path: str):
        cls.hps = get_hparams_from_file(config_path)
        cls.speaker_choices = list(
            map(lambda x: str(x[0]) + ":" + x[1], enumerate(cls.hps.speakers)))

        logger.info(
            f"config init done with config path {config_path}"
        )
