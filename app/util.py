
import json
import pathlib
# import tqdm

from typing import Optional
import os
import threading

from loguru import logger
# from app.common import HParams
# from __ini import HParams
from pathlib import Path
import requests

from app import HParams


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





# def download_defaults(model_path: pathlib.Path, config_path: pathlib.Path):

#     config = requests.get(config_url,  timeout=10).content
#     with open(str(config_path), 'wb') as f:
#         f.write(config)

#     t = threading.Thread(target=pdownload, args=(model_url, str(model_path)))
#     t.start()


def get_paths(dir_path: Path):

    model_path: Path = find_path_by_suffix(dir_path, "onnx")
    config_path: Path = find_path_by_suffix(dir_path, "json")
    # if not model_path or not config_path:
    #     model_path = dir_path / "model.onnx"
    #     config_path = dir_path / "config.json"
    #     logger.warning(
    #         "unable to find model or config, try to download default model and config"
    #     )
    #     download_defaults(model_path, config_path)

    # model_path = str(model_path)
    # config_path = str(config_path)
    # logger.info(f"model path: {model_path} config path: {config_path}")
    return model_path, config_path
