from typing import List, Union, Optional, Literal

import text
import torch
import torch.nn as nn

from utils import get_basic_config
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser

from .fastpitch.model import FastPitch as _FastPitch
from ..diacritizers import load_vowelizer

_VOWELIZER_TYPE = Literal['shakkala', 'shakkelha']

def text_collate_fn(batch: List[torch.Tensor]):
    """
    Args:
        batch: List[text_ids]
    Returns:
        text_ids_pad
        input_lens_sorted
        reverse_ids 
    """
    input_lens_sorted, input_sort_ids = torch.sort(
        torch.LongTensor([len(x) for x in batch]), descending=True)
    max_input_len = input_lens_sorted[0]

    text_ids_pad = torch.LongTensor(len(batch), max_input_len)
    text_ids_pad.zero_()
    for i in range(len(input_sort_ids)):
        text_ids = batch[input_sort_ids[i]]
        text_ids_pad[i, :text_ids.size(0)] = text_ids

    return text_ids_pad, input_lens_sorted, input_sort_ids.argsort()


class FastPitch(_FastPitch):
    def __init__(self,
                 checkpoint: str,         
                 arabic_in: bool = True,
                 vowelizer: Optional[_VOWELIZER_TYPE] = None,              
                 **kwargs):
        from models.fastpitch import net_config
        state_dicts = torch.load(checkpoint, map_location='cpu')
        if 'config' in state_dicts:
            net_config = state_dicts['config']
        super().__init__(**net_config)
        #self.n_eos = len(EOS_TOKENS)
        self.arabic_in = arabic_in

        #if checkpoint is not None:            
        self.load_state_dict(state_dicts['model'])

        self.config = get_basic_config()
        
        self.vowelizers = {}        
        if vowelizer is not None:
            self.vowelizers[vowelizer] = load_vowelizer(vowelizer, self.config)
        self.default_vowelizer = vowelizer

        self.phon_to_id = None
        if 'symbols' in state_dicts:
            self.phon_to_id = {phon: i for i, phon in enumerate(state_dicts['symbols'])}

        self.eval()
   
    @property
    def device(self):
        return next(self.parameters()).device

    def _vowelize(self, utterance: str, vowelizer: Optional[_VOWELIZER_TYPE] = None):
        vowelizer = self.default_vowelizer if vowelizer is None else vowelizer
        if vowelizer is not None:
            if not vowelizer in self.vowelizers:
                self.vowelizers[vowelizer] = load_vowelizer(vowelizer, self.config)
                # print(f"loaded: {vowelizer}")    
            utterance_ar = text.buckwalter_to_arabic(utterance)
            utterance = self.vowelizers[vowelizer].predict(utterance_ar)
        return utterance

    def _tokenize(self, utterance: str, vowelizer: Optional[_VOWELIZER_TYPE] = None):
        utterance = self._vowelize(utterance=utterance, vowelizer=vowelizer)
        if self.arabic_in:
            return text.arabic_to_tokens(utterance, append_space=False)
        return text.buckwalter_to_tokens(utterance, append_space=False)

    @torch.inference_mode()
    def ttmel_single(self,
                     utterance: str,
                     speed: float = 1,
                     speaker_id: int = 0,
                     vowelizer: Optional[_VOWELIZER_TYPE] = None
                     ):

        tokens = self._tokenize(utterance, vowelizer=vowelizer)

        token_ids = text.tokens_to_ids(tokens, self.phon_to_id)
        ids_batch = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
        sid = torch.LongTensor([speaker_id]).to(self.device)

        # Infer spectrogram     
        mel_spec, *_ = self.infer(ids_batch, 
                                  pace=speed, 
                                  speaker=speaker_id)

        mel_spec = mel_spec[0]

        return mel_spec  # [F, T]

    @torch.inference_mode()
    def ttmel_batch(self,
                    batch: List[str],
                    speed: float = 1,
                    speaker_id: int = 0,
                    vowelizer: Optional[_VOWELIZER_TYPE] = None
                    ):

        batch_tokens = [
            self._tokenize(line, vowelizer=vowelizer) 
            for line in batch
            ]

        batch_ids = [torch.LongTensor(
            text.tokens_to_ids(tokens, self.phon_to_id)
            ) for tokens in batch_tokens]

        batch = text_collate_fn(batch_ids)
        (
            batch_ids_padded, batch_lens_sorted,
            reverse_sort_ids
        ) = batch

        batch_ids_padded = batch_ids_padded.to(self.device)
        batch_lens_sorted = batch_lens_sorted.to(self.device)

        batch_sids = batch_lens_sorted*0 + speaker_id

        y_pred = self.infer(batch_ids_padded, pace=speed, speaker=speaker_id)     
        mel_outputs, mel_specgram_lengths, *_ = y_pred

        mel_list = []
        for i, id in enumerate(reverse_sort_ids):
            mel = mel_outputs[id, :, :mel_specgram_lengths[id]]
            mel_list.append(mel)

        return mel_list

    def ttmel(self,
              text_input: Union[str, List[str]],
              speed: float = 1,
              speaker_id: int = 0,
              batch_size: int = 1,
              vowelizer: Optional[_VOWELIZER_TYPE] = None
              ):
        # input: string
        if isinstance(text_input, str):
            return self.ttmel_single(text_input, speed=speed, 
                                     speaker_id=speaker_id,
                                     vowelizer=vowelizer)

        # input: list
        assert isinstance(text_input, list)
        batch = text_input
        mel_list = []

        if batch_size == 1:
            for sample in batch:
                mel = self.ttmel_single(sample, speed=speed, 
                                        speaker_id=speaker_id,
                                        vowelizer=vowelizer)
                mel_list.append(mel)
            return mel_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.ttmel_batch(batch, speed=speed, 
                                    speaker_id=speaker_id,
                                    vowelizer=vowelizer)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            mels = self.ttmel_batch(batch, speed=speed, 
                                    speaker_id=speaker_id,
                                    vowelizer=vowelizer)
            mel_list += mels

        return mel_list


class FastPitch2Wave(nn.Module):
    def __init__(self,
                 model_sd_path: str,
                 vocoder_sd: Optional[str] = None,
                 vocoder_config: Optional[str] = None,
                 vowelizer: Optional[_VOWELIZER_TYPE] = None,
                 arabic_in: bool = True,           
                 ):

        super().__init__()

        # from models.fastpitch import net_config
        state_dicts = torch.load(model_sd_path, map_location='cpu')
        # if 'config' in state_dicts:
        #     net_config = state_dicts['config']

        model = FastPitch(model_sd_path, 
                          arabic_in=arabic_in,
                          vowelizer=vowelizer)       
        model.load_state_dict(state_dicts['model'], strict=False)
        self.model = model

        if vocoder_sd is None or vocoder_config is None:
            config = get_basic_config()
            vocoder_sd = config.vocoder_state_path
            vocoder_config = config.vocoder_config_path

        vocoder = load_hifigan(vocoder_sd, vocoder_config)
        self.vocoder = vocoder
        self.denoiser = Denoiser(vocoder)

        self.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return x

    @torch.inference_mode()
    def tts_single(self,
                   text_buckw: str,
                   speed: float = 1,  
                   speaker_id: int = 0,
                   denoise: float = 0,   
                   vowelizer: Optional[_VOWELIZER_TYPE] = None,                             
                   return_mel=False):

        mel_spec = self.model.ttmel_single(text_buckw, speed, 
                                           speaker_id, vowelizer)
          
        wave = self.vocoder(mel_spec)

        if denoise > 0:
            wave = self.denoiser(wave, denoise)

        if return_mel:
            return wave[0].cpu(), mel_spec

        return wave[0].cpu()

    @torch.inference_mode()
    def tts_batch(self,
                  batch: List[str],
                  speed: float = 1,
                  speaker_id: int = 0,
                  denoise: float = 0,
                  vowelizer: Optional[_VOWELIZER_TYPE] = None,
                  return_mel: bool = False
                  ):

        mel_list = self.model.ttmel_batch(batch, speed, 
                                          speaker_id, vowelizer)

        wav_list = []
        for mel in mel_list:  
            wav_inferred = self.vocoder(mel)
            if denoise > 0:
                wav_inferred = self.denoiser(wav_inferred, denoise)

            wav_list.append(wav_inferred[0].cpu())

        if return_mel:
            wav_list, mel_list

        return wav_list

    def tts(self,
            text_input: Union[str, List[str]],
            speed: float = 1.,
            denoise: float = 0, 
            speaker_id: int = 0,
            batch_size: int = 2,
            vowelizer: Optional[_VOWELIZER_TYPE] = None,          
            return_mel: bool = False):
        """
        Args:
            text_input (str|List[str]): Input text.
            speed (float): Speaking speed.
            denoise (float): Hifi-GAN Denoiser strength.
            speaker_id (int): Speaker Id.
            batch_size (int): bacch size for inferrence.
            vowelizer (None|str): options [None, `'shakkala'`, `'shakkelha'`].
            return_mel (bool): Whether to return the mel spectrogram(s).
        """

        # input: string
        if isinstance(text_input, str):
            return self.tts_single(text_input, speaker_id=speaker_id,
                                   speed=speed, denoise=denoise,
                                   vowelizer=vowelizer,     
                                   return_mel=return_mel)

        # input: list
        assert isinstance(text_input, list)
        batch = text_input
        wav_list = []

        if batch_size == 1:
            for sample in batch:
                wav = self.tts_single(sample, speaker_id=speaker_id,
                                      speed=speed, denoise=denoise,
                                      vowelizer=vowelizer,
                                      return_mel=return_mel)
                wav_list.append(wav)
            return wav_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.tts_batch(batch, speaker_id=speaker_id,
                                  speed=speed, denoise=denoise,
                                  vowelizer=vowelizer,                  
                                  return_mel=return_mel)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            wavs = self.tts_batch(batch, speaker_id=speaker_id,
                                  speed=speed, denoise=denoise,
                                  vowelizer=vowelizer,                          
                                  return_mel=return_mel)
            wav_list += wavs

        return wav_list
    
