import torch
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import soundfile as sf

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import utils
import commons

from hparams import Hparams
from datasets import TextAudioSpeakerCollate, TextAudioSpeakerLoader, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from losses import discriminator_loss, kl_loss, generator_loss, feature_loss


def main():
    '''Assume Single Node Multi GPUs Training Only'''
    assert torch.cuda.is_available(), 'CPU training is not allowed.'
    
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8011'
    
    hps = Hparams()
    mp.spawn(run, nprocs = n_gpus, args = (n_gpus, hps,))
    
    
def run(rank, n_gpus, hps : Hparams):
    
    print('rank = ', rank)
    
    global global_steps
    
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = n_gpus, rank = rank)
    torch.manual_seed(hps.train_seed)
    torch.cuda.set_device(rank)
    
    train_dataset = TextAudioSpeakerLoader('./filelists/magicdata_audio_sid_train.16000.cleaned.txt', hps)
    train_sampler = DistributedBucketSampler(train_dataset, hps.batch_size,
                                             [32,300,400,500,600,700,800,900,1000],
                                             num_replicas = n_gpus, rank = rank, shuffle = True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers = 4, shuffle = False, pin_memory = True,
                              collate_fn = collate_fn, batch_sampler = train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader('./filelists/magicdata_audio_sid_valid.16000.cleaned.txt', hps)
        eval_loader = DataLoader(eval_dataset, num_workers = 4, shuffle = False,
                                 batch_size = hps.batch_size, pin_memory = True, 
                                 drop_last = False, collate_fn = collate_fn)
    
    
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
                           gin_channels = hps.gin_channels).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.learning_rate, betas = hps.betas, eps = hps.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.learning_rate, betas = hps.betas, eps = hps.eps)
    net_g = DDP(net_g, device_ids = [rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids = [rank], find_unused_parameters=True)
    
    
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, 'G_*.pth'), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, 'D_*.pth'), net_d, optim_d)
        global_steps = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_steps = 0
        
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma = hps.lr_decay, last_epoch = epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma = hps.lr_decay, last_epoch = epoch_str - 2)
    
    scaler = GradScaler(enabled = hps.fp16_run)
    
    for epoch in range(epoch_str, hps.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               scaler, [train_loader, eval_loader])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               scaler, [train_loader, None])
        scheduler_g.step()
        scheduler_d.step()
        
def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    
    train_loader.batch_sampler.set_epoch(epoch)
    global global_steps

    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking = True), x_lengths.cuda(rank, non_blocking = True)
        spec, spec_lengths = spec.cuda(rank, non_blocking = True), spec_lengths.cuda(rank, non_blocking = True)
        y, y_lengths = y.cuda(rank, non_blocking = True), y_lengths.cuda(rank, non_blocking = True)
        speakers = speakers.cuda(rank, non_blocking = True)
        
        time.sleep(1)
        
        with autocast(enabled = hps.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)
            
            mel = spec_to_mel_torch(spec, hps.filter_length, hps.n_mel_channels, hps.sampling_rate,
                                    hps.mel_fmin, hps.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.segment_size // hps.hop_length)
            
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.filter_length, hps.n_mel_channels,
                                              hps.sampling_rate, hps.hop_length, hps.win_length,
                                              hps.mel_fmin, hps.mel_fmax)
            y = commons.slice_segments(y, ids_slice * hps.hop_length, hps.segment_size)
            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled = False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        
        with autocast(enabled = hps.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled = False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        
        if rank == 0:
            if global_steps % hps.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                
                print('Train Epoch: %d [%.2f%%] G-Loss: %.4f D-Loss: %.4f Loss-g-fm: %.4f Loss-g-mel: %.4f Loss-g-dur: %.4f Loss-g-kl: %.4f lr: %.8f grad_norm_g: %.4f grad_norm_d: %.4f'%(epoch, 
                    100.*batch_idx/len(train_loader),
                    loss_gen_all, loss_disc_all, loss_fm, loss_mel, loss_dur, loss_kl,
                    lr, grad_norm_g, grad_norm_d))
                
            if global_steps % hps.eval_interval == 0:
                evaluate(hps, net_g, eval_loader)
                utils.save_checkpoint(net_g, optim_g, lr, epoch, os.path.join(hps.model_dir, 'G_{}.pth'.format(global_steps)))
                utils.save_checkpoint(net_d, optim_d, lr, epoch, os.path.join(hps.model_dir, 'D_{}.pth'.format(global_steps)))
        global_steps += 1
        
    if rank == 0:
        print('======> Epoch: {}'.format(epoch))
            
            
def evaluate(hps : Hparams, generator, eval_loader):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)
            
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            break
        
        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len = 1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.hop_length
        
        mel = spec_to_mel_torch(spec, hps.filter_length, hps.n_mel_channels, hps.sampling_rate,
                                hps.mel_fmin, hps.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), hps.filter_length, hps.n_mel_channels,
                                          hps.sampling_rate, hps.hop_length, hps.win_length,
                                          hps.mel_fmin, hps.mel_fmax)
        
        out_mel_filename = os.path.join(hps.eval_save_dir, 'gen_steps_%d_%d.mel.npy'%(global_steps, batch_idx))
        out_wav_filename = os.path.join(hps.eval_save_dir, 'gen_steps_%d_%d.wav'%(global_steps, batch_idx))
        np.save(out_mel_filename, y_hat_mel[0].cpu().numpy())
        sf.write(out_wav_filename, y_hat[0, :, :y_hat_lengths[0]].squeeze(0).cpu().numpy(), hps.sampling_rate)
        
        generator.train()
            
                
if __name__ == '__main__':
    main()        
        
        
