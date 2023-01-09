
import json
import pathlib
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


def time_it(func: callable):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        # print(f"func {func.__name__} cost {end-start} seconds")
        logger.info(f"func {func.__name__} cost {end-start} seconds")
        return res
    return wrapper


def download_defaults(model_path: pathlib.Path, config_path: pathlib.Path):
    import requests
    from tqdm import tqdm

    model_url = r"https://ccdesue-my.sharepoint.com/personal/ccds_ccdesue_onmicrosoft_com/_layouts/15/download.aspx?share=ET6NYdpJa7tBsM3vICCcqbsBpnEdRnnTq9m9P26ENutFVA"
    config_url = r"https://ccdesue-my.sharepoint.com/personal/ccds_ccdesue_onmicrosoft_com/_layouts/15/download.aspx?share=EdsTRVK3l-FJnKNt6fJ15-8BIvq_dsqMIJuAz29oEezqTg"

    
    @time_it
    def pdownload(url: str, save_path: str):
  
        file_size = int(requests.head(url).headers["Content-Length"])
        CHUNK_SIZE = 8192
        response = requests.get(url, stream=True)
        with tqdm(total=file_size, unit="B",
                  unit_scale=True, desc="progress") as pbar:

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(CHUNK_SIZE)

    pdownload(model_url, str(model_path))

    config = requests.get(config_url).content
    with open(str(config_path), 'wb') as f:
        f.write(config)


def get_paths() -> tuple[str, str]:
    dir_path = pathlib.Path(__file__).parent.absolute() / ".model"
    dir_path.mkdir(
        parents=True, exist_ok=True
    )

    model_path = find_path_by_suffix(dir_path, "onnx")
    config_path = find_path_by_suffix(dir_path, "json")
    if not model_path or not config_path:
        model_path = dir_path / "model.onnx"
        config_path = dir_path / "config.json"
        logger.warning(
            "unable to find model or config, try to download default model and config"
        )
        download_defaults(model_path, config_path)

    model_path = str(model_path)
    config_path = str(config_path)
    logger.info(f"model path: {model_path} config path: {config_path}")
    return model_path, config_path
