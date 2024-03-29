class Hparams():
    
    def __init__(self):
        
        # data
        self.text_cleaners = ['cn_cleaner2']
        self.max_wav_value = 32768.0
        self.sampling_rate = 16000
        self.filter_length = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = None
        self.add_blank = True
        self.n_speakers = 6
        self.cleaned_text = True
        self.min_text_len = 1
        self.max_text_len = 190
        
        self.symbol_to_id = '/ZFS4T/tts/data/Tacotron2_16k/symbol_to_id.json'
        self.model_dir = '/ZFS4T/tts/data/VITS/magicdata/model_saved/'
        self.eval_save_dir = '/ZFS4T/tts/data/VITS/magicdata/eval_saved/'
        
        # train
        self.train_seed = 1234
        self.n_vocab = 191
        self.segment_size = 8192
        self.inter_channels = 192
        self.hidden_channels = 192
        self.filter_channels = 768
        self.n_heads = 2
        self.n_layers = 6
        self.kernel_size = 3
        self.p_dropout = 0.1
        self.resblock = '1'
        self.resblock_kernel_size = [3,7,11]
        self.resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
        self.upsample_rates = [8,8,2,2]
        self.upsample_initial_channel = 512
        self.upsample_kernel_sizes = [16,16,4,4]
        self.n_layers_q = 3
        self.use_spectral_norm = False
        self.gin_channels = 256
        
        self.log_interval = 25
        self.eval_interval = 2000
        self.epochs = 10000
        self.learning_rate = 2e-4
        self.betas = [0.8, 0.99]
        self.eps = 1e-9
        self.batch_size = 24
        self.fp16_run = True
        self.lr_decay = 0.999875
        self.init_lr_ratio = 1
        self.warmup_epochs = 0
        self.c_mel = 45
        self.c_kl = 1.
        
        
        