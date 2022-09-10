import utils
import commons
from torch import LongTensor,no_grad,FloatTensor

from mel_processing import spectrogram_torch
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text

device = 'cpu'

class raw_vits():
    def __init__(self, config_path: str, model_path: str):
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.net_g_ms = SynthesizerTrn(
            len(self.hps_ms.symbols),
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers,
            **self.hps_ms.model)
        _ = self.net_g_ms.eval()
        utils.load_checkpoint(model_path, self.net_g_ms, None)
        
    def get_text(self, text: str, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, self.hps_ms.symbols, [])
        else:
            text_norm = text_to_sequence(text, self.hps_ms.symbols, self.hps_ms.data.text_cleaners)
        if self.hps_ms.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        global device 
        text_norm = LongTensor(text_norm).to(device)
        return text_norm
    
    def vc(self,src_id,target_id,audio):
        y = FloatTensor(audio).to(device)
        y = y.unsqueeze(0)
        spec = spectrogram_torch(y, self.hps_ms.data.filter_length,
                                self.hps_ms.data.sampling_rate, self.hps_ms.data.hop_length, self.hps_ms.data.win_length,
                                center=False).to(device)
        
        spec_lengths = LongTensor([spec.size(-1)]).to(device)
        sid_src = LongTensor([src_id]).to(device)
        sid_tgt = LongTensor([target_id]).to(device)
        with no_grad():
            audio = self.net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
            
        return audio

    def tts(self,cleanned_text_tensor,sid):
            with no_grad():
                x_tst = cleanned_text_tensor.unsqueeze(0)
                x_tst_lengths = LongTensor([cleanned_text_tensor.size(0)]).to(device)
                sid = LongTensor([sid]).to(device)
                audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            return audio
        
# print ("hello")