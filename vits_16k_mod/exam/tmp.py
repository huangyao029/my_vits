if __name__ == '__main__':
    lst = [
        '/mnt/disk3/huangyao/tts/vits/filelists/magicdata_audio_sid_train.16000.cleaned.txt',
        '/mnt/disk3/huangyao/tts/vits/filelists/magicdata_audio_sid_valid.16000.cleaned.txt'
    ]

    dit = {}
    idx = 0
    for filepth in lst:
        with open(filepth, 'r') as f:
            for line in f.readlines():
                for p in line.rstrip().split('|')[-1].split():
                    if p not in dit:
                        dit[p] = idx
                        idx += 1

    print(dit)