import onnxruntime






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
