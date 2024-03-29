import glob
import os
import random
import json

from parser_text_to_pyin import get_pyin



def main():
    
    data_dir = '/ZFS4T/opensrl/aishell1_part/'
    text_pth = '/ZFS4T/opensrl/aishell1_part/transcript/aishell_transcript_v0.8.txt'
    
    spk_to_id = {
        'S0176' : 0,
        'S0666' : 1,
        'S0710' : 2,
        'S0711' : 3
    }
    
    out_tr_text = '/mnt/disk3/huangyao/tts/vits/filelists/aishell_audio_sid_train.txt'
    out_tr_phone = '/mnt/disk3/huangyao/tts/vits/filelists/aishell_audio_sid_train.cleaned.txt'
    out_cv_text = '/mnt/disk3/huangyao/tts/vits/filelists/aishell_audio_sid_valid.txt'
    out_cv_phone = '/mnt/disk3/huangyao/tts/vits/filelists/aishell_audio_sid_valid.cleaned.txt'
    
    trans_dict = {}
    with open(text_pth, 'r') as f_trans:
        for line in f_trans.readlines():
            tmp = line.rstrip().split()
            basename = tmp[0]
            text = ''.join(tmp[1:])
            trans_dict[basename] = text
            
    with open(out_tr_text, 'w') as f_tr_text, open(out_tr_phone, 'w') as f_tr_phone,\
        open(out_cv_text, 'w') as f_cv_text, open(out_cv_phone, 'w') as f_cv_phone:
        for spk_name, spk_id in spk_to_id.items():
            wav_pth_list = glob.glob(os.path.join(data_dir, spk_name, '*.wav'))
            random.shuffle(wav_pth_list)
            wav_pth_list_len = len(wav_pth_list)
            
            for wav_pth in wav_pth_list[:int(wav_pth_list_len * 0.95)]:
                wav_basename = os.path.basename(wav_pth).split('.wav')[0]
                if wav_basename not in trans_dict:
                    print('%s not in trans_dict'%(wav_basename))
                    continue
                trans = trans_dict[wav_basename]
                f_tr_text.write('%s|%s|%s\n'%(wav_pth, str(spk_id), trans))
                pyin, _ = get_pyin(trans, False, False)
                f_tr_phone.write('%s|%s|%s\n'%(wav_pth, str(spk_id), pyin))
                
            for wav_pth in wav_pth_list[int(wav_pth_list_len * 0.95):]:
                wav_basename = os.path.basename(wav_pth).split('.wav')[0]
                if wav_basename not in trans_dict:
                    print('%s not in trans_dict'%(wav_basename))
                    continue
                trans = trans_dict[wav_basename]
                f_cv_text.write('%s|%s|%s\n'%(wav_pth, str(spk_id), trans))
                pyin, _ = get_pyin(trans, False, False)
                f_cv_phone.write('%s|%s|%s\n'%(wav_pth, str(spk_id), pyin))
                
                
def main_magicdata():
    info_file_path = '/ZFS4T/tts/data/MagicTTS_datasets/data_expand/info.json'
    data_root_dir = '/ZFS4T/tts/data/MagicTTS_datasets/data_expand/22050/'
    wave_kind = 'Wave-PC'
    
    out_tr_text = '/mnt/disk3/huangyao/tts/vits/filelists/magicdata_audio_sid_train.22050.txt'
    out_tr_phone = '/mnt/disk3/huangyao/tts/vits/filelists/magicdata_audio_sid_train.22050.cleaned.txt'
    out_cv_text = '/mnt/disk3/huangyao/tts/vits/filelists/magicdata_audio_sid_valid.22050.txt'
    out_cv_phone = '/mnt/disk3/huangyao/tts/vits/filelists/magicdata_audio_sid_valid.22050.cleaned.txt'
    
    with open(info_file_path, 'r') as f:
        info_dict = json.load(f)
        
    with open(out_tr_text, 'w') as ftrtext, open(out_tr_phone, 'w') as ftrphone,\
        open(out_cv_text, 'w') as fcvtext, open(out_cv_phone, 'w') as fcvphone:
        for spk_dir in glob.glob(os.path.join(data_root_dir, '*')):
            wav_paths = glob.glob(os.path.join(spk_dir, wave_kind, '*.wav'))
            for i, wav_path in enumerate(wav_paths):
                name = os.path.basename(wav_path).split('.wav')[0]
                cn_text = info_dict[name]['cn']
                spk_id = int(name[1:4]) - 1
                pyin, text = get_pyin(cn_text, False, True)
                if i < 195:
                    ftrtext.write('%s|%d|%s\n'%(wav_path, spk_id, text))
                    ftrphone.write('%s|%d|%s\n'%(wav_path, spk_id, pyin))
                else:
                    fcvtext.write('%s|%d|%s\n'%(wav_path, spk_id, text))
                    fcvphone.write('%s|%d|%s\n'%(wav_path, spk_id, pyin))
        
                
                
if __name__ == '__main__':
    #main()
    main_magicdata()
                
    
    