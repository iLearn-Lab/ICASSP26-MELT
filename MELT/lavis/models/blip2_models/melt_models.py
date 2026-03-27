"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from .create_diffusion import create_gaussian_diffusion
from lavis.diffusion_models.resample import create_named_schedule_sampler

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from sklearn.cluster import KMeans

from lavis.models.blip2_models.rarity_module import RarityFix



def l2norm(X, dim=-1):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def l1norm(X, dim):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    X = torch.div(X, norm)
    return 

def info_nce(query, target):
    bs = query.size(0)
    targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(query.device)
    temp = nn.Parameter(0.07 * torch.ones([]))
    x = torch.matmul(query,target).squeeze().to(query.device)
    #print('x',x.shape)
    sim_i2t,_ = x.max(-1)
    sim_i2t = sim_i2t / temp
    return F.cross_entropy(sim_i2t, targets)


@registry.register_model("Blip2QformerCir")
class MeltModel(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        diffusion=True,
        # num_diffusion_query=31

    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        self.diffusion_model = Diffusion(256) if diffusion else None

        # modification strength
        self.init_strength=torch.tensor(0.8)
        # strength=F.softplus(self.init_strength).clamp(0.1, 5)#约0.8
        strength= self.init_strength
        self.strength = nn.Parameter(torch.tensor(float(strength)))

        self.rarity_fix = RarityFix(dim=self.Qformer.config.hidden_size,
                                    num_heads=12, topk=5, strength=self.strength)
        self.token2vit = nn.Linear(self.Qformer.config.hidden_size,
                                   self.visual_encoder.num_features)  # 768 -> 1408
        nn.init.zeros_(self.token2vit.weight)
        nn.init.zeros_(self.token2vit.bias)


    def info_nce(self, query, target):
        sim_t2q = torch.matmul(
            query.unsqueeze(1).unsqueeze(1), target.permute(0, 2, 1)
        ).squeeze()
        bs = query.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(query.device)
        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / self.temp
        return F.cross_entropy(sim_i2t, targets)

    def forward(self, samples,device,idx=None, diffusion=None, schedule_sampler=None):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        image_embeds = self.ln_vision(self.visual_encoder(image))    #[B,S_img,d_img]
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)




        #(RATR）模块
        text_cls = self.forward_text(text_tokens)  # [B,d_q]
        image_token,_ = self.forward_image(image_embeds)

        # (RATR）识别并修改稀有语义
        #Rarity-Aware Token Refinement
        image_token = self.rarity_fix(text_cls, image_token)

        compose_high = self.token2vit(image_token)  # [B, 32, 1408]

        image_embeds = image_embeds + compose_high.mean(dim=1, keepdim=True)  # [B, 1, 1408] broadcast
        #
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )


        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        #Target fea
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )


        #fusion fea
        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )



        #diffusion loss   Diffusion-Based Similarity Denoising(DSD)
        pos = torch.ones((target_feats.shape[0], 1), dtype=torch.float)
        num_diffusion_query = target_feats.shape[0]-1
        neg = torch.zeros((target_feats.shape[0], num_diffusion_query), dtype=torch.float)
        micro = torch.cat([pos, neg], dim=1).to(target_feats.device)

        t, _ = schedule_sampler.sample(target_feats.shape[0], fusion_feats.device)

        diffusion_target_feat = target_feats.detach()
        diffusion_fusion_feat = fusion_feats.detach()

        sim_t2q_diff = torch.matmul(
            diffusion_fusion_feat.unsqueeze(1).unsqueeze(1), diffusion_target_feat.unsqueeze(0).transpose(2,3)
        ).squeeze(2)

        sim_i2t_diff, _ = sim_t2q_diff.max(-1)


        t2i_sim_logits=sim_i2t_diff
        t2i_sim_logits = F.softmax(t2i_sim_logits, dim=1)


        mask = torch.eq(idx.unsqueeze(0), idx.unsqueeze(0).t())
        t2i_sim_logits.masked_fill_(mask, -1)


        image_feat_neg = []
        for b in range(t2i_sim_logits.size(0)):
            _, neg_idx = t2i_sim_logits[b].topk(num_diffusion_query, largest=True, sorted=True)
            temp = [diffusion_target_feat[b]]
            for i in neg_idx:
                temp.append(diffusion_target_feat[i])
            image_feat_neg.append(torch.stack(temp, dim=0))
        image_feat_neg = torch.stack(image_feat_neg, dim=0)
        # print("image_feat_neg",image_feat_neg.shape)


        #train  denoiser
        output = diffusion.training_losses(self.diffusion_model,
                                           micro,
                                           t,
                                           {"fusion_feat": diffusion_fusion_feat,
                                            # "tar_image_feat": image_feat_neg.mean(2)},
                                           "tar_image_feat": image_feat_neg},
                                           temp=self.temp)

        #constraint loss
        loss_diffusion = output[("kl")]



        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()
        bs = fusion_feats.size(0)
        sim_i2t, _ = sim_t2q.max(-1)
        score_matrix_t2i = sim_i2t / self.temp

        # score_matrix_t2i=score_matrix_t2i.detach


        B, _ = sim_i2t.shape
        K = B-1
        device = score_matrix_t2i.device



        pos_idx = torch.arange(B, device=device)  # (B,)

        mask_diag = torch.eye(B, dtype=torch.bool, device=device)
        score_noPos = score_matrix_t2i.masked_fill(mask_diag, -1e4)
        _, neg_idx = score_noPos.topk(K, dim=1, largest=True, sorted=True)  # (B,K)


        col_idx = torch.cat([pos_idx[:, None], neg_idx], dim=1)
        row_idx = torch.arange(B, device=device)[:, None].expand(B, K + 1)


        micro = score_matrix_t2i[row_idx, col_idx]  # (B,1+K)

        S, dim = target_feats.shape[1], target_feats.shape[2]
        token_idx = torch.arange(S, device=device).view(1, 1, S).expand(B, K + 1, S)
        batch_idx = col_idx.unsqueeze(-1).expand(B, K + 1, S)  # (B,1+K,S)
        tar_stack = target_feats[batch_idx, token_idx]  # (B,1+K,S,dim)

        # denoising proces
        #Diffusion-Based Similarity Denoising
        sample = diffusion.ddim_sample_loop(
            self.diffusion_model,
            (B, K + 1),
            micro,  # 正例+K负例
            clip_denoised=True,
            model_kwargs={
                "fusion_feat": fusion_feats,  # (B,D)
                "tar_image_feat": tar_stack  # (B,1+K,S,D)
            }
        )


        score_new = score_matrix_t2i.clone()

        score_new[row_idx, col_idx] = sample.to(dtype=score_new.dtype)


        score_matrix_t2i = score_new

        T_diff = score_matrix_t2i.detach() / self.temp




        S_raw = sim_i2t / self.temp
        log_raw = F.log_softmax(S_raw, dim=1)
        S_diff = F.softmax(T_diff, dim=1)
        #knowledge distillation loss
        loss_KD = F.kl_div(log_raw, S_diff, reduction='batchmean')

        #Loss_bbc
        loss_bbc = self.info_nce(fusion_feats, target_feats)#Eq13

        # loss = loss_stu_rank
        loss = loss_bbc+0.3*loss_KD+ 0.1*loss_diffusion

        print("loss_stu_rank+loss_DistillKL+ loss_diffusion",loss_bbc,loss_KD,loss_diffusion)




        return {'loss_stu_rank':loss }

    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image_embeds):
        # image_embeds = self.ln_vision(self.visual_encoder(image))

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )


        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit


    
    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod, return_attns=False):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()

        # return image_embeds
        reference_embeds = image_embeds_frozen

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            mod,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        text_cls = self.forward_text(text_tokens)  # [B,d_q]
        image_token, _ = self.forward_image(reference_embeds)

        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        # return fusion_feats.unsqueeze(1).unsqueeze(1),R_score, is_rare
        return fusion_feats.unsqueeze(1).unsqueeze(1)

    @torch.no_grad()
    def extract_retrieval_target(self, img):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features.permute(0, 2, 1)


    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)



 #  diffusion process
class Diffusion(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, temp=1):
        super(Diffusion, self).__init__()
        self.width = embed_dim
        self.dropout = dropout
        self.temp = temp
        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.width, bias=False)
        self.v_proj = nn.Linear(self.width, self.width, bias=False)
        self.proj = nn.Linear(self.width, self.width)

        self.sequence_pos_encoder = PositionalEncoding(self.width, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.width, self.sequence_pos_encoder)

        self.decoder = nn.Sequential(
            nn.Linear(self.width * 2, self.width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.width * 2, 1),
        )

    def forward(self, x, timesteps, fusion_feat, tar_image_feat):
        B, N, S, D = tar_image_feat.shape

        cond_emb = self.embed_timestep(timesteps).squeeze(1)  # (B,dim)

        token_sim = torch.einsum('bd,bnsd->bns', [fusion_feat, tar_image_feat])  # (B,N,S)

        _, best_idx = token_sim.max(dim=-1)

        b_idx = torch.arange(B, device=tar_image_feat.device).view(B, 1).expand(B, N)
        n_idx = torch.arange(N, device=tar_image_feat.device).view(1, N).expand(B, N)
        tar_image_feat = tar_image_feat[b_idx, n_idx, best_idx]



        q = self.q_proj(fusion_feat + cond_emb)
        k = self.k_proj(tar_image_feat + cond_emb.unsqueeze(1))
        v = self.v_proj(tar_image_feat + cond_emb.unsqueeze(1))

        weight = torch.einsum('bd,bnd->bn', [q, k])
        weight = weight + x
        weight = torch.softmax(weight, dim=-1)
        new_emb = torch.einsum('bn,bnd->bd', [weight, v])
        new_emb = self.proj(new_emb)

        emb = torch.cat([new_emb.unsqueeze(1).repeat(1, tar_image_feat.shape[1], 1), tar_image_feat],
                        dim=-1)

        p = self.decoder(emb).squeeze(2)
        p += weight
        p,_=p.sort(dim=1,descending=True)

        return p


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])
