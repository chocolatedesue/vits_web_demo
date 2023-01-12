
import json
# from app.common import HParams
# from __ini import HParams
from pathlib import Path

from loguru import logger


# import tqdm


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

 

def get_paths(dir_path: Path):

    model_path: Path = find_path_by_suffix(dir_path, "onnx")
    config_path: Path = find_path_by_suffix(dir_path, "json")
    return model_path, config_path
