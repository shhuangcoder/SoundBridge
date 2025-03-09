# -*- coding: utf-8 -*-
import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import decord
import pandas as pd
import torch
from ipdb import set_trace
from decord import cpu
import cv2
import io,os
from .data_utils import video_loader
import torch.nn as nn

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform=None, is_training=False, tokenizer=None):
        ### common setups ###
        self.ego_root = cfg.ego_root
        self.exo_root = cfg.exo_root
        self.metadata = cfg.metadata
        
        self.clip_length = cfg.clip_length
        self.ctx_length = cfg.ctx_length
        
        ### maybe customized ###
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        with open(self.metadata, 'r') as f:
            self.samples = json.load(f)
        self.fps = -1
        
        all_scenes = [self.samples[str(i)]['query']['scene'] for i in range(len(self.samples))]
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_scenes)
        
    def __len__(self):
        return len(self.samples)
    
    
    def load_singleview_video(self, option, view):
        epsilon = 1e-6
        if view == 'ego':
            frames = self.transform(video_loader(self.ego_root, option['video_uid'], 
                                    float(option['start_sec']), end_second=float(option['end_sec']),
                                    clip_length=self.clip_length, fps=self.fps,
                                    jitter=self.is_training))
            
            env_features = torch.zeros((1,256))

            audio_root = '/path/to/audio_beats/'
            audio_feature_path = osp.join(audio_root, f"{option['video_uid']}_{int(option['start_sec'])}-{int(option['end_sec'])}.pt")   

            if osp.exists(audio_feature_path):
                audio_features = torch.load(audio_feature_path)
                audio_features = audio_features[0].unsqueeze(0)
                audio_features = audio_features.transpose(1, 2)
                audio_features = torch.mean(audio_features, -1)
            else:
                feature_dim = 768
                audio_features = torch.zeros(1, feature_dim)
                
            audio_features += epsilon

        else:
            frames = self.transform(video_loader(self.exo_root, option['video_uid'], 
                                    float(option['start_sec']), end_second=float(option['end_sec']),
                                    clip_length=self.clip_length, fps=self.fps,
                                    jitter=self.is_training))

            env_features = torch.zeros((1,256))

            audio_root = '/path/to/audio_beats/'
            audio_feature_path = osp.join(audio_root, f"{option['video_uid']}_{int(option['start_sec'])}-{int(option['end_sec'])}.pt")   

            if osp.exists(audio_feature_path):
                audio_features = torch.load(audio_feature_path)
                audio_features = audio_features[0].unsqueeze(0)
                audio_features = audio_features.transpose(1, 2)
                audio_features = torch.mean(audio_features, -1)
            else:
                feature_dim = 768
                audio_features = torch.zeros(1, feature_dim)

            audio_features += epsilon
        
        return frames, env_features, audio_features
    
    def get_raw_item_v2v(self, i):
        itemMCQ = self.samples[str(i)]
        answerIndex = itemMCQ['answer']
        videoQuery = itemMCQ['query']
        cur_type = itemMCQ['types']
        frameQuery, env_features, audio_features = self.load_singleview_video(videoQuery, 'ego' if cur_type == 1 else 'exo')
        textQuery = videoQuery['narration_en']
        textQuery_audio = videoQuery['narration_en_audio']
        scene = videoQuery['scene']
        scene = self.label_encoder.transform([scene])[0] 
        frames_options = []
        narration_options = []
        narration_options_audio = []
        env_features_options = []
        sampleOptions = itemMCQ['choices']
        audio_features_options = []
        for option_id in range(len(sampleOptions)):
            option = sampleOptions[str(option_id)]
            frames, env_features, audio_features = self.load_singleview_video(option, 'exo' if cur_type == 1 else 'ego')
            frames_options.append(frames)
            narration_options.append(option['narration_en'])
            narration_options_audio.append(option['narration_en_audio'])
            env_features_options.append(env_features)
            audio_features_options.append(audio_features)
        
        return frameQuery, textQuery, frames_options, narration_options, textQuery_audio, narration_options_audio, answerIndex, itemMCQ['types'], env_features, env_features_options, audio_features, audio_features_options, scene

    def __getitem__(self, i):
        frameQuery, textQuery, frames_options, narration_options, textQuery_audio, narration_options_audio, answerIndex, q_type, env_features, env_features_options, audio_features, audio_features_options, scene = self.get_raw_item_v2v(i)
        second_ids = 0

        raw_textQuery = textQuery
        raw_textQuery_audio = textQuery_audio
        raw_narration_options = narration_options
        raw_narration_options_audio = narration_options_audio
        
        if self.tokenizer is not None:
            raw_textQuery = self.tokenizer(raw_textQuery)
            raw_textQuery_audio = self.tokenizer(raw_textQuery_audio)
            raw_narration_options = self.tokenizer(raw_narration_options)
            raw_narration_options_audio = self.tokenizer(raw_narration_options_audio)
    
        frames = frames_options
        return frameQuery, torch.stack(frames, dim=0), answerIndex, q_type, raw_textQuery, raw_narration_options, raw_textQuery_audio, raw_narration_options_audio, i, env_features, env_features_options, audio_features, audio_features_options, scene
