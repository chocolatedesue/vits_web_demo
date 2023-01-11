import os
from pathlib import Path

from loguru import logger
# from app import CONFIG_URL, MODEL_URL
from app.util import get_hparams_from_file, get_paths, time_it
import requests
from tqdm.auto import tqdm
import re
from re import Pattern
import onnxruntime as ort
# import threading


MODEL_URL = r"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJmckZWcGdCR0xxLWJmU28/root/content"
CONFIG_URL = r"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhNEJ3enhhUHpqNE5EZWc/root/content"


class Config:
    hps: dict = None
    # pattern: Pattern = None
    # symbol_to_id:dict = None
    speaker_choices: list = None
    ort_sess: ort.InferenceSession = None
    model_is_ok: bool = False

    @classmethod
    @logger.catch
    @time_it
    def init(cls):

        # logger.add(
        #     "vits_infer.log",  rotation="10 MB", encoding="utf-8", enqueue=True, retention="30 days"
        # )

        dir_path = Path(__file__).parent.absolute() / ".model"
        dir_path.mkdir(
            parents=True, exist_ok=True
        )
        model_path, config_path = get_paths(dir_path)

        if not model_path or not config_path:
            model_path = dir_path / "model.onnx"
            config_path = dir_path / "config.json"
            logger.warning(
                "unable to find model or config, try to download default model and config"
            )
            cfg = requests.get(CONFIG_URL,  timeout=5).content
            with open(str(config_path), 'wb') as f:
                f.write(cfg)
            cls.setup_config(str(config_path))
            # import threading
            # t = threading.Thread(target=cls.pdownload,
            #                      args=(MODEL_URL, str(model_path)))
            # t.start()
            import multiprocessing

            p = multiprocessing.Process(target=cls.pdownload,
                                        args=(MODEL_URL, str(model_path)))
            p.start()
            # cls.pdownload(MODEL_URL, str(model_path))

        else:
            cls.setup_config(str(config_path))
            cls.setup_model(str(model_path))

    @classmethod
    @time_it
    def setup_model(cls, model_path: str):

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]


        cls.ort_sess = ort.InferenceSession(model_path, providers=providers)
        cls.model_warm_up(cls.ort_sess, cls.hps)
        cls.model_is_ok = True

        logger.info(
            f"model init done with model path {model_path}"
        )

    @classmethod
    def setup_config(cls, config_path: str):
        cls.hps = get_hparams_from_file(config_path)
        cls.speaker_choices = list(
            map(lambda x: str(x[0])+":"+x[1], enumerate(cls.hps.speakers)))

        logger.info(
            f"config init done with config path {config_path}"
        )

    @staticmethod
    def model_warm_up(ort_sess, hps):
        # init the model
        import numpy as np
        from .text import text_to_seq
        # seq = np.random.randint(low=0, high=len(
        #     hps.symbols), size=(1, 10), dtype=np.int64)

        seq = np.array(
            [text_to_seq("こにちわ、あやせです", hps=Config.hps)], dtype=np.int64)
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

    @classmethod
    @time_it
    def pdownload(cls, url: str, save_path: str, chunk_size: int = 8192):
        # copy from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_requests.py
        file_size = int(requests.head(url).headers["Content-Length"])
        response = requests.get(url, stream=True)
        with tqdm(total=file_size,  unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc="model download") as pbar:

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(chunk_size)
        cls.setup_model(save_path)
