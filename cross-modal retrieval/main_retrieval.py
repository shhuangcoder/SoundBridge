# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
import os
import pandas as pd
import sys
import time
import pickle
import random 

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.parallel
import wandb
from omegaconf import OmegaConf
from copy import deepcopy
import numpy as np
import logging

from models.builder import build_model

from models import model_utils
from models.tokenizer import generate_tokenizer
from function.meter import AverageMeter, ProgressMeter
from function import distributed as dist_utils
from function.utils import build_train_loader, build_val_loader, build_optimizer, resume_checkpoint, build_scheduler
from function.config import get_config
from function.logger import get_logger
from sklearn.metrics import recall_score, f1_score

def get_args_parser():
    parser = argparse.ArgumentParser(description='EgoExoLearn Association training and evaluation', add_help=False)
    # Data
    parser.add_argument('--config', default='configs/default.yml', type=str)

    # System
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
   
    parser.add_argument('--testonly', action='store_true', help='whether to perform test only')
    return parser


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main(args):
    ### Prepare env ###
    cfg = get_config(args)
    os.makedirs(cfg.output, exist_ok=True)
    
    dist_utils.init_distributed_mode(args)
    logger = get_logger(cfg)   
    ### save config file ###
    if dist_utils.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config save to {path}')

    ### log config ###
    logger.info(OmegaConf.to_yaml(cfg))

    global best_acc1
    random_seed(cfg.train.seed, dist_utils.get_rank())
    logger.info(f'Creating model:{cfg.model.name}')
    model = build_model(cfg.model)
    
    if cfg.model.freeze_temperature:
        logger.info('Freeze logit temperature')
        if hasattr(model, 'logit_scale'):
            model.logit_scale.requires_grad = False

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            # find_unused_parameters=cfg.train.find_unused_parameters
            find_unused_parameters=True,
        )
    tokenizer = generate_tokenizer(cfg.model.name)    

    criterion = model_utils.get_loss(cfg.model.name, args, cfg, tokenizer=tokenizer).cuda(args.gpu)
    optimizer = build_optimizer(cfg.train, model, criterion)
    scaler = amp.GradScaler(enabled=not cfg.train.disable_amp)
    lr_schedule = build_scheduler(cfg)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    loaded_resume = resume_checkpoint(cfg, model, optimizer, scaler, criterion)
    start_epoch, best_acc1 = loaded_resume['start_epoch'], loaded_resume['best_acc1']
    cudnn.benchmark = True
    
    logger.info("=> creating dataset")
    train_loader, train_sampler = build_train_loader(args, cfg, tokenizer)
    egobridge_v2v_loader = build_val_loader(args, cfg, dataset_name='egobridge_v2v', tokenizer=deepcopy(tokenizer))
    
    
    if dist_utils.is_main_process() and cfg.wandb:
        wandb_id = os.path.split(cfg.output)[-1]
        wandb.init(project='egoexo', id=wandb_id, config=args, resume='allow')

    if cfg.test.testonly:    
        ### V2V ###
        metrics = validate_v2v_mcq(egobridge_v2v_loader, model, use_half=False, cfg=cfg, args=args, logger=logger)
        
        logger.info('Cross-Modal Ego2Exo:Text->Video top1: {:.3f} | Exo2Ego:Text->Video top1: {:.3f}'.format(metrics['Ego2Exo:Text->Video_top1'], metrics['Exo2Ego:Text->Video_top1']))
        logger.info('Cross-Modal Ego2Exo:Text->Video top5: {:.3f} | Exo2Ego:Text->Video top5: {:.3f}'.format(metrics['Ego2Exo:Text->Video_top5'], metrics['Exo2Ego:Text->Video_top5']))
        logger.info('Cross-Modal Ego2Exo:Text->Video top10: {:.3f} | Exo2Ego:Text->Video top10: {:.3f}'.format(metrics['Ego2Exo:Text->Video_top10'], metrics['Exo2Ego:Text->Video_top10']))
        
        logger.info('Cross-Modal Ego2Exo:Video->Text top1: {:.3f} | Exo2Ego:Video->Text top1: {:.3f}'.format(metrics['Ego2Exo:Video->Text_top1'], metrics['Exo2Ego:Video->Text_top1']))
        logger.info('Cross-Modal Ego2Exo:Video->Text top5: {:.3f} | Exo2Ego:Video->Text top5: {:.3f}'.format(metrics['Ego2Exo:Video->Text_top5'], metrics['Exo2Ego:Video->Text_top5']))
        logger.info('Cross-Modal Ego2Exo:Video->Text top10: {:.3f} | Exo2Ego:Video->Text top10: {:.3f}'.format(metrics['Ego2Exo:Video->Text_top10'], metrics['Exo2Ego:Video->Text_top10']))
        exit(0)

    best_metric = 0.
    print("=> beginning training")
    for epoch in range(start_epoch, cfg.train.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger)
        
        ### logging training stats ###
        for k, v in train_stats.items():
            logger.info(f'Epoch {epoch}: Train_{k}: {round(v, 6)}')

        ### saving per epoch model ckpt before evaluation ###
        logger.info('=> saving per-epoch checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, False, cfg.output, is_epoch=True)

        logger.info('=> 0-shot on MCQ')
        v2v_metrics = validate_v2v_mcq(egobridge_v2v_loader, model, use_half=False, cfg=cfg, args=args,epoch=epoch, logger=logger)
        
        logger.info('Cross-Modal Ego2Exo:Text->Video top1: {:.3f} | Exo2Ego:Text->Video top1: {:.3f}'.format(v2v_metrics['Ego2Exo:Text->Video_top1'], v2v_metrics['Exo2Ego:Text->Video_top1']))
        logger.info('Cross-Modal Ego2Exo:Text->Video top5: {:.3f} | Exo2Ego:Text->Video top5: {:.3f}'.format(v2v_metrics['Ego2Exo:Text->Video_top5'], v2v_metrics['Exo2Ego:Text->Video_top5']))
        logger.info('Cross-Modal Ego2Exo:Text->Video top10: {:.3f} | Exo2Ego:Text->Video top10: {:.3f}'.format(v2v_metrics['Ego2Exo:Text->Video_top10'], v2v_metrics['Exo2Ego:Text->Video_top10']))
        
        logger.info('Cross-Modal Ego2Exo:Video->Text top1: {:.3f} | Exo2Ego:Video->Text top1: {:.3f}'.format(v2v_metrics['Ego2Exo:Video->Text_top1'], v2v_metrics['Exo2Ego:Video->Text_top1']))
        logger.info('Cross-Modal Ego2Exo:Video->Text top5: {:.3f} | Exo2Ego:Video->Text top5: {:.3f}'.format(v2v_metrics['Ego2Exo:Video->Text_top5'], v2v_metrics['Exo2Ego:Video->Text_top5']))
        logger.info('Cross-Modal Ego2Exo:Video->Text top10: {:.3f} | Exo2Ego:Video->Text top10: {:.3f}'.format(v2v_metrics['Ego2Exo:Video->Text_top10'], v2v_metrics['Exo2Ego:Video->Text_top10']))
        
        avg_map = 0.5 * (v2v_metrics['Ego->Exo_top1'] + v2v_metrics['Exo->Ego_top1'])
        
        if avg_map > best_metric:
            is_best = True
            best_metric = avg_map
        else:
            is_best = False   

        ### save checkpoint ###
        is_epoch = ((epoch + 1) % cfg.train.save_freq) == 0

        if args.distributed and cfg.train.use_zero:
            logger.info("=> consolidating state_dict before saving (due to ZeRO)")
            optimizer.consolidate_state_dict()

        logger.info('=> saving the best checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, is_best, cfg.output, is_epoch=is_epoch)


def train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = model_utils.get_metric_names(cfg)
    
    iters_per_epoch = len(train_loader) // cfg.train.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // cfg.train.update_freq
                
        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]
        
        batch_size = inputs['text'].size(0)

        model_inputs = [inputs['video'].cuda(args.gpu), inputs['text'].cuda(args.gpu), inputs['text_aug'].cuda(args.gpu), inputs['text_audio'].cuda(args.gpu), inputs['audio_features'].cuda(args.gpu), inputs['env_features'].cuda(args.gpu)]
        
        # compute output
        with amp.autocast(enabled=not cfg.train.disable_amp):
            outputs = model(
                *model_inputs,
                use_checkpoint=cfg.train.use_checkpoint,
                norm_embed=cfg.model.norm_embed
            )
            loss_dict = criterion(outputs)
            retrieval_loss = loss_dict['loss']
            retrieval_loss /= cfg.train.update_freq

            loss =  retrieval_loss  
            
        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % cfg.train.update_freq != 0:
            continue

        if cfg.train.clip_grad_value is not None:
            scaler.unscale_(optimizer)
            if cfg.train.clip_grad_type == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.clip_grad_value, norm_type=2.
                )
            elif cfg.train.clip_grad_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.train.clip_grad_value)
            else:
                assert False, f"Unknown clip mode ({cfg.train.clip_grad_type})."
        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        ### adjust logit scale ###
        if hasattr(dist_utils.get_model(model), 'logit_scale'):
            # clamp logit scale to [0, 100]
            dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
        else:
            logit_scale = torch.nan

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), cfg.train.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % cfg.train.print_freq == 0:
            if dist_utils.is_main_process():
                train_iter_log = {
                            'iter': data_iter,
                            'total_loss': round(loss.item(), 3),
                            **{k: round(v.item(), 3) for k, v in loss_dict.items()},
                           'scaler': round(scaler.get_scale(), 3), 
                           'logit': round(logit_scale, 3)}
                train_iter_log_str = ''
                for logk, logv in train_iter_log.items():
                    train_iter_log_str += f'{logk}:{logv}  '

                logger.info(train_iter_log_str)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}

def validate_v2v_mcq(val_loader, model, use_half=False, cfg=None, args=None, epoch=None, logger=None):
    model.eval()
    n = 0
    if use_half:
        model.half()

    all_types = []
    all_video_embeddings = []
    all_labels = []
    all_ground_truth_embeddings = []
    all_ground_truth_embeddings_crossModal = []
    
    with torch.no_grad():
        print('=> start forwarding')
        
        total_correct_scene = 0
        total_samples_scene = 0
        
        all_preds = []
        all_gts = []
        all_uids = []
        all_scene_labels = []
        all_pre_scenes = []
        all_scene_embeddings = []
        all_preds_crossModal_T2V = []
        all_preds_crossModal_V2T = []
        all_preds_crossModal_A2T = []
        
        end_time = time.time()
        
        for i, inputs in enumerate(val_loader):
            if i % 10 == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            
            frame_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            env_features = inputs[9].cuda(non_blocking=True)
            env_features = torch.mean(env_features, dim=1)
            audio_features = inputs[11].cuda(non_blocking=True)
            audio_features = torch.mean(audio_features, dim=1)
            
            scene_labels = inputs[13].cuda(non_blocking=True)
            all_scene_labels.append(scene_labels)
            
            if isinstance(inputs[10], list):
                env_features_options = torch.stack([item.clone().detach() for item in inputs[10]])
            else:
                env_features_options = inputs[10].clone().detach()
            env_features_options = env_features_options.cuda(non_blocking=True)
            
            if isinstance(inputs[12], list):
                audio_features_options = torch.stack([item.clone().detach() for item in inputs[12]])
            else:
                audio_features_options = inputs[12].clone().detach()
            audio_features_options = audio_features_options.cuda(non_blocking=True)
            
            if use_half:
                frames_options = frames_options.half()
                env_features_options = env_features_options.half()
                audio_features_options = audio_features_options.half()
        
            answer = inputs[2]
            q_type = inputs[3]
            uid = inputs[8]
            
            batch_size = frames_options.shape[0]
            frames_options = frames_options.view(-1, *frames_options.shape[2:])
            
            text = inputs[4]
            if isinstance(inputs[4], list):
                text = torch.stack([item.clone().detach() for item in inputs[4]])
            else:
                text = inputs[4].clone().detach()
            text = text.cuda(non_blocking=True)
            
            text_options = inputs[5]
            if isinstance(inputs[5], list):
                text_options_features = torch.stack([item.clone().detach() for item in inputs[5]])
            else:
                text_options_features = inputs[5].clone().detach()
            text_options_features = text_options_features.cuda(non_blocking=True)
            text_options_features = text_options_features.view(-1, *text_options_features.shape[2:])
            
            text_audio = inputs[6]
            if isinstance(inputs[6], list):
                text_audio = torch.stack([item.clone().detach() for item in inputs[6]])
            else:
                text_audio = inputs[6].clone().detach()
            text_audio = text_audio.cuda(non_blocking=True)
            
            text_options_audio = inputs[7]
            if isinstance(inputs[7], list):
                text_options_features_audio = torch.stack([item.clone().detach() for item in inputs[7]])
            else:
                text_options_features_audio = inputs[7].clone().detach()
            text_options_features_audio = text_options_features_audio.cuda(non_blocking=True)
            text_options_features_audio = text_options_features_audio.view(-1, *text_options_features_audio.shape[2:])
            
            audio_features_options = audio_features_options.view(-1, *audio_features_options.shape[2:])
            audio_features_options = torch.mean(audio_features_options, dim=1)
            
            ### encode texts ###
            text_features, audio_features = dist_utils.get_model(model).encode_text(text, text_audio, audio_features)
            text_options_features, audio_features_options = dist_utils.get_model(model).encode_text(text_options_features, text_options_features_audio, audio_features_options)
            text_options_features = text_options_features.view(batch_size, 20, -1)
        
            env_features_options = env_features_options.view(-1, *env_features_options.shape[2:])
            env_features_options = torch.mean(env_features_options, dim=1)
            
            ### encode videos ###
            image_query_features, scene_query_features = dist_utils.get_model(model).encode_image(frame_query, text_features, audio_features, env_features)
            image_options_features, _ = dist_utils.get_model(model).encode_image(frames_options, text_options_features, audio_features_options, env_features_options)
            image_options_features = image_options_features.view(batch_size, 20, -1)
            
            audio_features_options = audio_features_options.view(batch_size, 20, -1)

            all_types.append(q_type)
            all_video_embeddings.append(image_query_features.cpu())
            all_labels.append(q_type.cpu())

            all_gts.append(answer)
            all_uids.append(uid)
            all_scene_embeddings.append(scene_query_features.cpu())
            
            for j in range(batch_size):
                
                similarity_matrix_crossModal_T2V = torch.matmul(text_features[j], image_options_features[j].mT)
                similarity_matrix_crossModal_T2V = similarity_matrix_crossModal_T2V.cpu().detach()
                all_preds_crossModal_T2V.append(similarity_matrix_crossModal_T2V)
                
                similarity_matrix_crossModal_V2T = torch.matmul(image_query_features[j], text_options_features[j].mT)
                similarity_matrix_crossModal_V2T = similarity_matrix_crossModal_V2T.cpu().detach()
                all_preds_crossModal_V2T.append(similarity_matrix_crossModal_V2T)
                
                similarity_matrix_crossModal_A2T = torch.matmul(audio_features[j], text_options_features[j].mT)
                similarity_matrix_crossModal_A2T = similarity_matrix_crossModal_A2T.cpu().detach()
                all_preds_crossModal_A2T.append(similarity_matrix_crossModal_A2T)

                # 保存每个query的ground truth特征
                ground_truth_index = answer[j].item()
                ground_truth_embedding = image_options_features[j, ground_truth_index, :].cpu()
                all_ground_truth_embeddings.append(ground_truth_embedding)
                
                ground_truth_embedding_crossModal = text_options_features[j, ground_truth_index, :].cpu()
                all_ground_truth_embeddings_crossModal.append(ground_truth_embedding_crossModal)
                
            softmax_probs = torch.softmax(scene_query_features, dim=1)
            predicted_scene = torch.argmax(softmax_probs, dim=1)
            all_pre_scenes.append(predicted_scene)
            correct_predictions_scene = (predicted_scene == scene_labels).sum().item()
            total_correct_scene += correct_predictions_scene
            total_samples_scene += scene_labels.size(0)

        all_uids = torch.cat(all_uids)
        all_preds_crossModal_T2V = torch.stack(all_preds_crossModal_T2V)
        all_preds_crossModal_V2T = torch.stack(all_preds_crossModal_V2T)
        all_preds_crossModal_A2T = torch.stack(all_preds_crossModal_A2T)
        all_gts = torch.cat(all_gts)
        all_types = torch.cat(all_types)
        all_scene_labels = torch.cat(all_scene_labels)
        all_pre_scenes = torch.cat(all_pre_scenes)
        
        save_pred_results(all_uids, all_preds_crossModal_T2V, all_preds_crossModal_V2T, all_preds_crossModal_A2T, all_gts, all_scene_labels, cfg)
        
        metrics = egomcq_accuracy_metrics(all_preds_crossModal_T2V, all_preds_crossModal_V2T, all_preds_crossModal_A2T, all_gts, all_types, all_scene_labels)
        print(metrics)
        return metrics

def save_pred_results(uids, preds, preds_crossModal_T2V, preds_crossModal_V2T, preds_crossModal_A2T, gts, scene_labels, cfg):
    import pandas as pd
    all_data = []
    predictions = torch.max(preds, 1)[1]
    predictions_crossModal_T2V = torch.max(preds_crossModal_T2V, 1)[1]
    predictions_crossModal_V2T = torch.max(preds_crossModal_V2T, 1)[1]
    predictions_crossModal_A2T = torch.max(preds_crossModal_A2T, 1)[1]
    
    for i in range(len(uids)):
        uid = int(uids[i].cpu().numpy())
        prediction = int(predictions[i].cpu().numpy())
        prediction_crossModal_T2V = int(predictions_crossModal_T2V[i].cpu().numpy())
        prediction_crossModal_V2T = int(predictions_crossModal_V2T[i].cpu().numpy())
        prediction_crossModal_A2T = int(predictions_crossModal_A2T[i].cpu().numpy())
        gt = int(gts[i].cpu().numpy())
        scene_label = int(scene_labels[i].cpu().numpy())
        all_data.append([uid, prediction, prediction_crossModal_T2V, prediction_crossModal_V2T, prediction_crossModal_A2T, gt, scene_labels])
    
    df = pd.DataFrame(all_data, columns=['uid', 'pred', 'pred_crossModal_T2V', 'pred_crossModal_V2T', 'pred_crossModal_A2T', 'GT', 'scene_label'])
    csv_output_path = os.path.join(cfg.output, 'pred.csv')
    df.to_csv(csv_output_path, index=0)

def egomcq_accuracy_metrics(preds, preds_crossModal_T2V, preds_crossModal_V2T, preds_crossModal_A2T, labels, types, scene_labels):
    metrics = {}
    type_list = torch.unique(types)
    
    for type_i, group_i in zip(type_list, group_list):
        for top_k in top_k_list:
            correct = 0
            total = 0
            for pred, label, typer in zip(preds, labels, types):
                if typer == type_i:
                    top_k_preds = torch.topk(pred, top_k, dim=0).indices
                    if label.item() in top_k_preds.tolist():
                        correct += 1
                    total += 1
            accuracy = correct / total
            metrics[f'{group_i}_top{top_k}'] = accuracy * 100
            
    group_list_crossmodal_T2V = ['Ego2Exo:Text->Video', 'Exo2Ego:Text->Video']
    for type_i, group_i in zip(type_list, group_list_crossmodal_T2V):
        for top_k in top_k_list:
            correct_crossmodal_T2V = 0
            total_crossmodal_T2V = 0
            for pred_cross_modal_T2V, label, typer in zip(preds_crossModal_T2V, labels, types):
                if typer == type_i:
                    top_k_preds_crossmodal_T2V = torch.topk(pred_cross_modal_T2V, top_k, dim=0).indices
                    if label.item() in top_k_preds_crossmodal_T2V.tolist():
                        correct_crossmodal_T2V += 1
                    total_crossmodal_T2V += 1
            accuracy_crossmodal_T2V = correct_crossmodal_T2V / total_crossmodal_T2V
            metrics[f'{group_i}_top{top_k}'] = accuracy_crossmodal_T2V * 100
    
    group_list_crossmodal_V2T = ['Ego2Exo:Video->Text', 'Exo2Ego:Video->Text']
    for type_i, group_i in zip(type_list, group_list_crossmodal_V2T):
        for top_k in top_k_list:
            correct_crossmodal_V2T = 0
            total_crossmodal_V2T = 0
            for pred_cross_modal_V2T, label, typer in zip(preds_crossModal_V2T, labels, types):
                if typer == type_i:
                    top_k_preds_crossmodal_V2T = torch.topk(pred_cross_modal_V2T, top_k, dim=0).indices
                    if label.item() in top_k_preds_crossmodal_V2T.tolist():
                        correct_crossmodal_V2T += 1
                    total_crossmodal_V2T += 1
            accuracy_crossmodal_V2T = correct_crossmodal_V2T / total_crossmodal_V2T
            metrics[f'{group_i}_top{top_k}'] = accuracy_crossmodal_V2T * 100
            
    group_list_crossmodal_A2T = ['Ego2Exo:Audio->Text', 'Exo2Ego:Audio->Text']
    for type_i, group_i in zip(type_list, group_list_crossmodal_A2T):
        for top_k in top_k_list:
            correct_crossmodal_A2T = 0
            total_crossmodal_A2T = 0
            for pred_cross_modal_A2T, label, typer in zip(preds_crossModal_A2T, labels, types):
                if typer == type_i:
                    top_k_preds_crossmodal_A2T = torch.topk(pred_cross_modal_A2T, top_k, dim=0).indices
                    if label.item() in top_k_preds_crossmodal_A2T.tolist():
                        correct_crossmodal_A2T += 1
                    total_crossmodal_A2T += 1
            accuracy_crossmodal_A2T = correct_crossmodal_A2T / total_crossmodal_A2T
            metrics[f'{group_i}_top{top_k}'] = accuracy_crossmodal_A2T * 100
            
            
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EgoExoLearn Association training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
