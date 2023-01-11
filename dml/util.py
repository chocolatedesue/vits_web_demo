import json
from pathlib import Path
from typing import Optional
import onnxruntime as ort
from loguru import logger
import numpy as np

def time_it(func: callable):
    import time

    def wrapper(*args, **kwargs):
        # start = time.time()
        start = time.perf_counter()
        res = func(*args, **kwargs)
        # end = time.time()
        end = time.perf_counter()
        # print(f"func {func.__name__} cost {end-start} seconds")
        logger.info(f"func {func.__name__} cost {end-start} seconds")
        return res
    return wrapper
    
@time_it
def ort_infer(ort_sess: ort.InferenceSession, ort_inputs: dict):
    audio = np.squeeze(ort_sess.run(None, ort_inputs))
    audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
    audio: np.array = np.clip(audio, -32767.0, 32767.0)
    return audio


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def find_path_by_suffix(dir_path: Path, suffix: Path):
    assert dir_path.is_dir()

    for path in dir_path.glob(f"*.{suffix}"):
        return path

    return None


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result





def get_paths(dir_path: Path):

    model_path: Path = find_path_by_suffix(dir_path, "onnx")
    config_path: Path = find_path_by_suffix(dir_path, "json")
    return model_path, config_path


def model_warm_up(ort_sess, hps):
    # init the model
    import numpy as np
    seq = np.random.randint(low=0, high=len(
        hps.symbols), size=(1, 10), dtype=np.int64)

    # seq_len = torch.IntTensor([seq.size(1)]).long()
    seq_len = np.array([seq.shape[1]], dtype=np.int64)

    # noise(可用于控制感情等变化程度) lenth(可用于控制整体语速) noisew(控制音素发音长度变化程度)
    # 参考 https://github.com/gbxh/genshinTTS
    # scales = torch.FloatTensor([0.667, 1.0, 0.8])
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    # make triton dynamic shape happy
    # scales = scales.unsqueeze(0)
    scales.resize(1, 3)
    # sid = torch.IntTensor([0]).long()
    sid = np.array([0], dtype=np.int64)
    # sid = torch.LongTensor([0])
    ort_inputs = {
        'input': seq,
        'input_lengths': seq_len,
        'scales': scales,
        'sid': sid
    }
    ort_sess.run(None, ort_inputs)
