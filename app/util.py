
import json
from typing import Optional
import os

from loguru import logger 
from app.common import HParams


def find_path_by_postfix(dir_path: Optional[str], postfix: Optional[str]):

    if not os.path.exists(dir_path):
        return None
    assert isinstance(
        dir_path, str), f"dir_path must be str, but got {type(dir_path)}"
    assert isinstance(
        postfix, str), f"postfix must be str, but got {type(postfix)}"
    for i in os.listdir(dir_path):
        res = i.split('.')[-1]
        if res == postfix:
            return os.path.join(dir_path, i)
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
