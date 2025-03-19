# -*- coding: utf-8 -*-
from .openai_model import QuickGELU, Transformer
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .timesformer import SpaceTimeTransformer
from .openai_clip import load as load_openai_clip
from .model_utils import remap_keys
from copy import deepcopy
import torchvision.transforms as T


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model
        self.attn_mask = self.build_attention_mask()
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.attn_mask,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.ffn = nn.Linear(embed_dim, embed_dim, bias=True)
        self.ffn2 = nn.Linear(768, embed_dim, bias=True)
        
        self.cross_attn_image = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1)
        self.cross_attn_text = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1)
        
        self.cross_attn_fusion_visual = nn.TransformerEncoderLayer(
            d_model=embed_dim,   
            nhead=4,             
            dim_feedforward=embed_dim * 4,  
            dropout=0.1,         
            activation="gelu"    
        )
        
        self.cross_attn_fusion_text = nn.TransformerEncoderLayer(
            d_model=embed_dim,   
            nhead=4,             
            dim_feedforward=embed_dim * 4,  
            dropout=0.1,         
            activation="gelu"    
        )
        
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        
    def inflate_positional_embeds(self, curr_frames):
        '''
        # positional_embed: [self.ctx_length, D]
        # 
        # '''
        if self.context_length == curr_frames:
            return self.positional_embedding, self.attn_mask
        if self.context_length > curr_frames:
            return self.positional_embedding[:, :curr_frames, :], self.build_attention_mask(curr_frames)
        if self.context_length < curr_frames:
            new_temporal_embed = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), (curr_frames, self.positional_embedding.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            return torch.nn.Parameter(new_temporal_embed).to(self.positional_embedding.device), self.build_attention_mask(curr_frames)

    def build_attention_mask(self, ctx_length=None):
        if ctx_length is None:
            mask = torch.empty(self.context_length, self.context_length)
        else:
            mask = torch.empty(ctx_length, ctx_length)

        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    
    def normalize(self, tensor):
        min_val, _ = tensor.min(dim=1, keepdim=True)
        max_val, _ = tensor.max(dim=1, keepdim=True)
        return (tensor - min_val) / (max_val - min_val + 1e-8)
    
    def standardize(self, tensor):
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        return (tensor - mean) / (std + 1e-8)
    
    def encode_image(self, image, text, audio_features, env_features, use_checkpoint=False, apply_project=True):
        
        x = self.visual(image, use_checkpoint=use_checkpoint) 
        
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x
        
        x = x @ self.image_projection  

        audio_embed = audio_features.unsqueeze(0)
        image_embed = x.unsqueeze(0)
        visual_audio_embed, _ = self.cross_attn_image(query=audio_embed, key=image_embed, value=image_embed)
        visual_audio_embed = self.cross_attn_fusion_visual(visual_audio_embed)
        visual_audio_embed = visual_audio_embed.sequeeze(0)
        
        scene_x = visual_audio_embed

        return visual_audio_embed, scene_x

    def encode_text(self, text, text_audio, audio, use_checkpoint=False):
        x1 = self.token_embedding(text)  
        curr_ctx_len1 = x1.shape[1]
        positional_embedding1, attn_mask1 = self.inflate_positional_embeds(curr_ctx_len1)
        x1 = x1 + positional_embedding1
        x1 = x1.permute(1, 0, 2)  # NLD -> LND
        x1 = self.transformer(x1, use_checkpoint=use_checkpoint, attn_mask=attn_mask1)
        x1 = x1.permute(1, 0, 2)  # LND -> NLD
        x1 = self.ln_final(x1)
        x1 = x1[torch.arange(x1.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        x2 = self.token_embedding(text_audio)
        curr_ctx_len2 = x2.shape[1]
        positional_embedding2, attn_mask2 = self.inflate_positional_embeds(curr_ctx_len2)
        x2 = x2 + positional_embedding2
        x2 = x2.permute(1, 0, 2)  # NLD -> LND
        x2 = self.transformer(x2, use_checkpoint=use_checkpoint, attn_mask=attn_mask2)
        x2 = x2.permute(1, 0, 2)  # LND -> NLD
        x2 = self.ln_final(x2)
        x2 = x2[torch.arange(x2.shape[0]), text_audio.argmax(dim=-1)] @ self.text_projection
        
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        visual_audio_embed, _ = self.cross_attn_text(query=x2, key=x1, value=x1)
        visual_audio_embed = self.cross_attn_fusion_text(visual_audio_embed)
        visual_audio_embed = visual_audio_embed.sequeeze(0)
        x1 = x1.squeeze(0)
        x2 = x2.squeeze(0)

        audio_features = x2
        
        return visual_audio_embed, audio_features
        

    def forward(self, image, text, text_aug, text_audio, audio_features, env_features, use_checkpoint=False, norm_embed=False):
        env_features = torch.mean(env_features, dim=1)
        
        audio_features = self.ffn2(audio_features)
        audio_features = torch.mean(audio_features, dim=1)
        audio_features = self.standardize(audio_features)

        text_embed, audio_embed = self.encode_text(text, text_audio, audio_features, use_checkpoint=use_checkpoint)
        image_embed, scene_embed = self.encode_image(image, text_embed, audio_features, env_features, use_checkpoint=use_checkpoint)
        
        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
            audio_embed = F.normalize(audio_embed, dim=-1)
            scene_embed = F.normalize(scene_embed, dim=-1)
            
        outputs = {'image_embed': image_embed,
                'text_embed': text_embed,
                'audio_embed': audio_embed,
                'scene_embed': scene_embed,
                'logit_scale': self.logit_scale.exp()}
        
        return outputs

def CLIP_OPENAI_TIMESFORMER_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0.0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=512, freeze_text_encoder=False, pretrained_visual_checkpoint=None,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    # print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    # print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))
    
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=16,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    
    
    if freeze_text_encoder:
        print("Freeze Text Encoder!!!")
        for module in [model.token_embedding, model.positional_embedding, model.transformer, model.ln_final, model.text_projection]:
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for p in module.parameters():
                    p.requires_grad=False
          
    return model

