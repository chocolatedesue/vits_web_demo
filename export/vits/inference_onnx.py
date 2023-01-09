# Copyright (c) 2022, Yongqiang Li (yongqiangli@alumni.hust.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from text import text_to_sequence
import numpy as np
from scipy.io import wavfile
import torch
import json
import commons
import utils
import sys
import pathlib

try:
    import onnxruntime as ort
except ImportError:
    print('Please install onnxruntime!')
    sys.exit(1)


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.detach().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--onnx_model', required=True, help='onnx model')
    parser.add_argument('--cfg', required=True, help='config file')
    parser.add_argument('--outdir', default="onnx_output",
                        help='ouput directory')
    # parser.add_argument('--phone_table',
    #                     required=True,
    #                     help='input phone dict')
    # parser.add_argument('--speaker_table', default=None, help='speaker table')
    parser.add_argument('--test_file', required=True, help='test file')
    args = parser.parse_args()
    return args


def get_symbols_from_json(path):
    import os
    assert os.path.isfile(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data['symbols']


def main():
    args = get_args()
    print(args)
    if not pathlib.Path(args.outdir).exists():
        pathlib.Path(args.outdir).mkdir(exist_ok=True, parents=True)
    # phones =
    symbols = get_symbols_from_json(args.cfg)
    phone_dict = {
        symbol: i for i, symbol in enumerate(symbols)
    }

    # speaker_dict = {}
    # if args.speaker_table is not None:
    #     with open(args.speaker_table) as p_f:
    #         for line in p_f:
    #             arr = line.strip().split()
    #             assert len(arr) == 2
    #             speaker_dict[arr[0]] = int(arr[1])
    hps = utils.get_hparams_from_file(args.cfg)

    ort_sess = ort.InferenceSession(args.onnx_model)

    with open(args.test_file) as fin:
        for line in fin:
            arr = line.strip().split("|")
            audio_path = arr[0]
        
            # TODO: 控制说话人编号
            sid = 0
            text = arr[1]
            # else:
            #     sid = speaker_dict[arr[1]]
            #     text = arr[2]
            seq = text_to_sequence(text, symbols=hps.symbols, cleaner_names=["japanese_cleaners2"]
                                   )
            if hps.data.add_blank:
                seq = commons.intersperse(seq, 0)

            # if hps.data.add_blank:
            #     seq = commons.intersperse(seq, 0)
            with torch.no_grad():
                #         x = torch.LongTensor([seq])
                #         x_len = torch.IntTensor([x.size(1)]).long()
                #         sid = torch.LongTensor([sid]).long()
                #         scales = torch.FloatTensor([0.667, 1.0, 1])
                # # make triton dynamic shape happy
                #         scales = scales.unsqueeze(0)

                # use numpy to replace torch
                x = np.array([seq], dtype=np.int64)
                x_len = np.array([x.shape[1]], dtype=np.int64)
                sid = np.array([sid], dtype=np.int64)
                scales = np.array([0.667, 1.0, 1], dtype=np.float32)
                # scales = scales[np.newaxis, :]
                # scales.reshape(1, -1)
                scales.resize(1, 3)

                ort_inputs = {
                    'input': x,
                    'input_lengths': x_len,
                    'scales': scales,
                    'sid': sid
                }

                # ort_inputs = {
                #     'input': to_numpy(x),
                #     'input_lengths': to_numpy(x_len),
                #     'scales': to_numpy(scales),
                #     'sid': to_numpy(sid)
                # }
                import time
                start_time = time.time()

                audio = np.squeeze(ort_sess.run(None, ort_inputs))
                audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
                audio = np.clip(audio, -32767.0, 32767.0)

                end_time = time.time()
                print("time cost: ", end_time - start_time)

                wavfile.write(args.outdir + "/" + audio_path.split("/")[-1],
                              hps.data.sampling_rate, audio.astype(np.int16))


if __name__ == '__main__':
    main()
