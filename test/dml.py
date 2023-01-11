import util
from scipy.io import wavfile
import time
import onnxruntime as ort
# import loguru
from loguru import logger
import numpy as np

logger.info(
    ort.get_available_providers()
)
provider = ["DmlExecutionProvider"]

so = ort.SessionOptions()
# For CPUExecutionProvider you can change it to True
so.enable_mem_pattern = False

ort_sess = ort.InferenceSession(
    "/home/ccds/func/vits_web_demo/app/.model/model.onnx", providers=provider, sess_options=so)

txt = [0, 1, 1, 1]
seq = np.array([txt], dtype=np.int64)

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


# from util import model_pre_activate

util.model_pre_activate(ort_sess)
# start_time = time.time()
start_time = time.perf_counter()
audio = np.squeeze(ort_sess.run(None, ort_inputs))
audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
audio = np.clip(audio, -32767.0, 32767.0)
end_time = time.perf_counter()
# end_time = time.time()
print("infer time cost: ", end_time - start_time, "s")
sample_rate = 22050


wavfile.write("test.wav", sample_rate, audio.astype(np.int16))
