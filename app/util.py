
import json
from typing import Optional
import os

from loguru import logger 
from app.common import HParams
from pathlib import Path

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

def time_it(func:callable):
    import time
    def wrapper(*args,**kwargs):
        start = time.time()
        res = func(*args,**kwargs)
        end = time.time()
        # print(f"func {func.__name__} cost {end-start} seconds")
        logger.info(f"func {func.__name__} cost {end-start} seconds")
        return res
    return wrapper
