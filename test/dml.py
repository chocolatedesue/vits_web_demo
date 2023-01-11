import util
# from scipy.io import wavfile
import time
import onnxruntime as ort
# import loguru
from pydub import AudioSegment
from loguru import logger
import numpy as np

from text import text_to_seq, get_hparams_from_file

MODEL_PATH = r"C:\Users\ccds\Desktop\model.onnx"
CONFIG_PATH = r"C:\Users\ccds\Desktop\config.json"
logger.info(
    ort.get_available_providers()
)
provider = ["DmlExecutionProvider"]

so = ort.SessionOptions()
# For CPUExecutionProvider you can change it to True
so.enable_mem_pattern = False

ort_sess = ort.InferenceSession(
    MODEL_PATH, providers=provider, sess_options=so)
hps = get_hparams_from_file(CONFIG_PATH)
txt = text_to_seq("あまりよく覚えていないようねこれはお守りではありません! 責任です", hps)
seq = np.array([txt], dtype=np.int64)

seq_len = np.array([seq.shape[1]], dtype=np.int64)
scales = np.array([0.667, 1.2, 0.8], dtype=np.float32)
scales.resize(1, 3)
sid = np.array([0], dtype=np.int64)
ort_inputs = {
    'input': seq,
    'input_lengths': seq_len,
    'scales': scales,
    'sid': sid
}

# from util import model_pre_activate

util.model_warm_up(ort_sess, hps)
# start_time = time.time()
# start_time = time.perf_counter()

# end_time = time.perf_counter()
# # end_time = time.time()
# print("infer time cost: ", end_time - start_time, "s")
sample_rate = 22050
audio = util.ort_infer(
    ort_sess, ort_inputs
)

# wavfile.write("test.wav", sample_rate, audio.astype(np.int16))
audio = AudioSegment(
    audio.astype(np.int16).tobytes(),
    frame_rate=22050,
    sample_width=2,
    channels=1
)

# save the AudioSegment object as a WAV file
audio.export("example.wav", format="wav")
