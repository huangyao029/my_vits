import torch
import os
import json
import tqdm
import time

import soundfile as sf
import numpy as np

from hparams import Hparams
from models import SynthesizerTrn
from parser_text_to_pyin import get_pyin
from utils import load_checkpoint
from commons import intersperse
from mel_processing import spectrogram_torch


with open('./symbol_to_id.json', 'r') as f:
    symbol_to_id_dict = json.load(f)


def text_to_speech(text_list, spk_id, model_idx, out_wav_dir, hps : Hparams, 
                   noise_scale = 1., length_scale = 1., noise_scale_w = 1.):
    
    # 模型
    net_g = SynthesizerTrn(n_vocab = hps.n_vocab, spec_channels = hps.filter_length // 2 + 1,
                           segment_size = hps.segment_size // hps.hop_length,
                           inter_channels = hps.inter_channels, 
                           hidden_channels = hps.hidden_channels,
                           filter_channels = hps.filter_channels,
                           n_heads = hps.n_heads, n_layers = hps.n_layers, kernel_size = hps.kernel_size,
                           p_dropout = hps.p_dropout, resblock = hps.resblock, 
                           resblock_kernel_sizes = hps.resblock_kernel_size,
                           resblock_dilation_sizes = hps.resblock_dilation_sizes,
                           upsample_rates = hps.upsample_rates, 
                           upsample_initial_channel = hps.upsample_initial_channel,
                           upsample_kernel_sizes = hps.upsample_kernel_sizes, n_speakers = hps.n_speakers,
                           gin_channels = hps.gin_channels).cuda()
    model_path = os.path.join(hps.model_dir, 'G_%d.pth'%(model_idx))
    _ = net_g.eval()
    _ = load_checkpoint(model_path, net_g, None)
    
    with torch.no_grad():
        # 文字 -> 拼音（韵母声母分开）(基于大辞典)
        for i, text in tqdm.tqdm(enumerate(text_list)):
            start_text_to_pyin = time.time()
            pyin, _  = get_pyin(text, False, True)
            pyin_seq = [symbol_to_id_dict[x] for x in [['_'] + pyin.rstrip().split() + ['~']][0]]
            if hps.add_blank:
                pyin_seq = intersperse(pyin_seq, 0)
            pyin_seq = torch.LongTensor(pyin_seq)
            pyin_seq = pyin_seq.cuda().unsqueeze(0)
            pyin_seq_lengths = torch.LongTensor([pyin_seq.size(1)]).cuda()
            end_text_to_pyin = time.time()
            spk_id_ = torch.LongTensor([spk_id]).cuda()
            audio = net_g.infer(x = pyin_seq, x_lengths = pyin_seq_lengths, sid = spk_id_, noise_scale = noise_scale,
                                length_scale = length_scale, noise_scale_w = noise_scale_w)[0][0, 0].data.cpu().float().numpy()
            audio = audio / (np.max(np.abs(audio))) * 0.7
            end_pyin_to_wav = time.time()
            out_wav_pth = os.path.join(out_wav_dir, 'vits_multispeaker_infer_modelidx-%d_spkid-%d_wavid-%d_ns-%.2f_ls-%.2f_nsw-%.2f.wav'%(model_idx, 
                                                                                                            spk_id, i, noise_scale,
                                                                                                            length_scale,
                                                                                                            noise_scale_w))
            sf.write(out_wav_pth, audio, hps.sampling_rate)
            print('text length -> %d words, text -> pyin : %.2f ms, pyin -> wav : %.2f ms'%(len(text), 
                                                                                            (end_text_to_pyin - start_text_to_pyin) * 1000,
                                                                                            (end_pyin_to_wav - end_text_to_pyin) * 1000))
        
        
        
def voice_conversion(src_wav_pth, src_spk_id, target_spk_id, model_idx, out_wav_dir, hps : Hparams, 
                     noise_scale = 1., length_scale = 1., noise_scale_w = 1.):
    
    # 模型
    net_g = SynthesizerTrn(n_vocab = hps.n_vocab, spec_channels = hps.filter_length // 2 + 1,
                           segment_size = hps.segment_size // hps.hop_length,
                           inter_channels = hps.inter_channels, 
                           hidden_channels = hps.hidden_channels,
                           filter_channels = hps.filter_channels,
                           n_heads = hps.n_heads, n_layers = hps.n_layers, kernel_size = hps.kernel_size,
                           p_dropout = hps.p_dropout, resblock = hps.resblock, 
                           resblock_kernel_sizes = hps.resblock_kernel_size,
                           resblock_dilation_sizes = hps.resblock_dilation_sizes,
                           upsample_rates = hps.upsample_rates, 
                           upsample_initial_channel = hps.upsample_initial_channel,
                           upsample_kernel_sizes = hps.upsample_kernel_sizes, n_speakers = hps.n_speakers,
                           gin_channels = hps.gin_channels).cuda()
    model_path = os.path.join(hps.model_dir, 'G_%d.pth'%(model_idx))
    _ = net_g.eval()
    _ = load_checkpoint(model_path, net_g, None)
    
    with torch.no_grad():
        start_read_wav = time.time()
        audio, samplerate = sf.read(src_wav_pth)
        audio = torch.FloatTensor(audio.astype(np.float32))
        audio = audio.unsqueeze(0)
        spec = spectrogram_torch(audio, hps.filter_length, hps.sampling_rate, 
                                hps.hop_length, hps.win_length, center = False).cuda()
        end_spec = time.time()
        spec_length = torch.LongTensor([spec.size(2)]).cuda()
        src_spk_id_ = torch.LongTensor([src_spk_id]).cuda()
        target_spk_id_ = torch.LongTensor([target_spk_id]).cuda()
        audio_vc = net_g.voice_conversion(spec, spec_length, sid_src = src_spk_id_,
                                          sid_tgt = target_spk_id_)[0][0, 0].data.cpu().float().numpy()
        end_conversion = time.time()
        audio_vc = audio_vc / (np.max(np.abs(audio_vc))) * 0.7
        out_wav_pth = os.path.join(out_wav_dir, 'vits_voice_conversion_modelidx-%d_srcspkid-%d_tgtspkid-%d.wav'%(model_idx, src_spk_id, target_spk_id))
        sf.write(out_wav_pth, audio_vc, hps.sampling_rate)
        
        print('audio length : %.2f s, source speaker id : %d, target speaker id : %d. audio -> spec : %.2f ms, voice conversion : %.2f ms'%(
            audio.size(1) / hps.sampling_rate, src_spk_id, target_spk_id, (end_spec - start_read_wav) * 1000,
            (end_conversion - end_spec) * 1000
        ))
        
        
        
        
if __name__ == '__main__':
    
    # text_list = [
    #     '一个正在路上行走的银行家，觉得这个行当不行。',
    #     '一般不会吧，老板出货都是攒的旧货。',
    #     '别人的场，人不够了，帮吆喝下。',
    #     '贴地飞行比真正的飞行要危险。',
    #     '西边北边爬山啊，平路没啥意思。',
    #     '看设计，当然碳布选择也是设计的一部分。',
    #     '数字只是数字，生命更重要。',
    #     '在一个迷人的森林中，有一只小松鼠名叫小橙，他梦想着成为一名勇敢的探险家。有一天，他踏上了旅程，遇到了一只友好的小兔子，他们一起经历了惊险刺激的冒险，最终实现了自己的梦想。'
    # ]
    
    text_list = [
        '一个正在路上行走的银行家，觉得这个行当不行。'
    ]

    spk_id = 5
    model_idx = 244000
    out_wav_dir = '/ZFS4T/tts/data/VITS/magicdata/test_saved/'
    hps = Hparams()
    noise_scale = 1.
    length_scale = 1.
    noise_scale_w = 1.
    text_to_speech(text_list, spk_id, model_idx, out_wav_dir, hps, 
                   noise_scale = noise_scale, length_scale = length_scale,
                   noise_scale_w = noise_scale_w)
              
                   
    # src_spk_id = 0
    # target_spk_id = 5
    # model_idx = 244000
    # out_wav_dir = '/ZFS4T/tts/data/VITS/magicdata/test_saved/'
    # src_wav_pth = '/ZFS4T/tts/data/MagicTTS_datasets/data_expand/16000/F001/Wave-PC/F001_188.wav'
    # hps = Hparams()
    # voice_conversion(src_wav_pth = src_wav_pth, src_spk_id = src_spk_id, target_spk_id = target_spk_id,
    #                  model_idx = model_idx, out_wav_dir = out_wav_dir, hps = hps)
    
    
        
    
    
    
    