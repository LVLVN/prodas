from torch import nn
from timm.layers import trunc_normal_
import torch
import torch.nn.functional as F
from typing import Optional, List
from einops.layers.torch import Rearrange
from einops import rearrange
import clip

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.neg_slope)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

class ConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        return x + self.conv(x)


class ContextGate(nn.Module):
    def __init__(self):
        super(ContextGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        return x * self.conv(feat)


class DynModulator(nn.Module):
    def __init__(self, dim, reduction=8):
        super(DynModulator, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid_dim = max(dim // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class AttnRefiner(nn.Module):
    def __init__(self, dim):
        super(AttnRefiner, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x.unsqueeze(2)
        x2 = x.unsqueeze(2)
        feat = torch.cat([x1, x2], dim=2)
        feat = Rearrange('b c t h w -> b (c t) h w')(feat)
        return x * self.conv(feat)


class MAFRM(nn.Module):
    def __init__(self, in_dim, out_dim=None, reduction=8, enable_scm=True, enable_cem=True, enable_arm=True):
        super(MAFRM, self).__init__()
        self.enable_scm = enable_scm
        self.enable_cem = enable_cem
        self.enable_arm = enable_arm
        out_dim = out_dim or in_dim
        
        self.adjust = None if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, 1, bias=True)
        
        if enable_scm:
            self.scm = ContextGate()
        if enable_cem:
            self.cem = DynModulator(out_dim if self.adjust else in_dim, reduction)
        if enable_arm:
            self.arm = AttnRefiner(out_dim if self.adjust else in_dim)
            
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim if self.adjust else in_dim, out_dim, 1, bias=True),
            nn.BatchNorm2d(out_dim)
        )
    
    def forward(self, x):
        if self.adjust is not None:
            x = self.adjust(x)
        
        identity = x
        
        if self.enable_scm:
            x = self.scm(x)
        if self.enable_cem:
            x = self.cem(x)
        if self.enable_arm:
            x = self.arm(x)
        
        return self.fusion(x) + identity


class MorphologicalAdaptationModule(nn.Module):
    def __init__(self, feature_channels, mask_channels=1, intermediate_channels=32):
        super(MorphologicalAdaptationModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_channels + mask_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.deformation_predictor = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Tanh()
        )
        self.blending_gate = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image_features, initial_mask):
        combined_input = torch.cat([image_features, initial_mask], dim=1)
        encoded_features = self.encoder(combined_input)
        deformation_field = self.deformation_predictor(encoded_features)
        deformed_mask = torch.clamp(initial_mask + deformation_field, 0, 1)
        gate = self.blending_gate(encoded_features)
        refined_mask = gate * deformed_mask + (1 - gate) * initial_mask
        return refined_mask

class VisionLanguageAttentiveMasker(nn.Module):
    def __init__(self, clip_dim=512, decoder_channels=(256, 128, 64)):
        super().__init__()
        
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.ModuleList()
        in_channels = clip_dim
        for out_channels in decoder_channels:
            self.decoder.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.decoder.append(nn.BatchNorm2d(out_channels))
            self.decoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.mask_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, image_features, text_features, sentence_features=None):
        image_features_norm = F.normalize(image_features, p=2, dim=-1)
        text_features_norm = F.normalize(text_features, p=2, dim=-1)
        
        similarity = torch.bmm(image_features_norm, text_features_norm.transpose(1, 2))
        
        patch_scores, _ = torch.max(similarity, dim=-1)
        
        patch_scores = 1.0 - patch_scores
        
        if sentence_features is not None:
            sentence_features_norm = F.normalize(sentence_features, p=2, dim=-1)
            sentence_similarity = torch.bmm(image_features_norm, sentence_features_norm.unsqueeze(-1))

            sentence_similarity = 1.0 - sentence_similarity

            B, N, _ = sentence_similarity.shape
            grid_size = int(N**0.5)
            sentence_similarity_map = sentence_similarity.squeeze(-1).reshape(B, 1, grid_size, grid_size)
            patch_scores_map = patch_scores.reshape(B, 1, grid_size, grid_size)
            
            fusion_input = torch.cat([patch_scores_map, sentence_similarity_map], dim=1)
            beta_map = self.fusion_gate(fusion_input)
            semantic_map = beta_map * patch_scores_map + (1 - beta_map) * sentence_similarity_map
        else:
            B, N = patch_scores.shape
            grid_size = int(N**0.5)
            semantic_map = patch_scores.reshape(B, 1, grid_size, grid_size)

        C = image_features.shape[-1]
        image_features_2d = image_features.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        
        weighted_features = image_features_2d * semantic_map

        x = weighted_features
        for layer in self.decoder:
            x = layer(x)
        
        mask = self.mask_head(x)
        return torch.sigmoid(mask)


class FactorizedSemanticAlignment(nn.Module):
    def __init__(self, text_dim, image_dim, num_concepts=64, dropout=0.1):
        super().__init__()
    
        assert text_dim == image_dim, "FSA requires text_dim and image_dim to be the same."
        inner_dim = image_dim

        self.num_concepts = num_concepts

        self.conceptual_bottleneck = nn.Parameter(torch.randn(num_concepts, inner_dim))
        
        self.to_q = nn.Identity()
        self.to_k = nn.Identity()
        self.to_v = nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, image_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(image_dim)

    def forward(self, image_feat, text_feat):
        q = self.to_q(image_feat)
        k = self.to_k(text_feat)
        v = self.to_v(text_feat)

        attn_img_to_concept = F.softmax(q @ self.conceptual_bottleneck.t(), dim=-1)

        attn_text_to_concept = F.softmax(k @ self.conceptual_bottleneck.t(), dim=-1)
        
        value_concepts = torch.bmm(attn_text_to_concept.transpose(1, 2), v)

        out = torch.bmm(attn_img_to_concept, value_concepts)
        
        enhanced_image_feat = self.layer_norm(image_feat + self.to_out(out))
        return enhanced_image_feat, attn_img_to_concept, attn_text_to_concept


class ProDASBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, up=False, down=False, fuse=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        if fuse:
            self.adjust = MAFRM(in_channels, out_channels)
        else:
            self.adjust = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        self.conv = ConvBlock(out_channels, dropout)
        if up:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels//2, 2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.LeakyReLU(0.1, inplace=True)
            )
        if down:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels*2, 2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels*2),
                nn.LeakyReLU(0.1, inplace=True)
            )
    
    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.adjust(x)
        x = self.conv(x)
        if not self.up and not self.down:
            return x
        elif self.up and not self.down:
            return x, self.upsample(x)
        elif not self.up and self.down:
            return x, self.downsample(x)
        else:
            return x, self.upsample(x), self.downsample(x)


class ProDAS(nn.Module):
    def __init__(self,  num_classes=1, num_channels=1, feature_scale=2, dropout=0.2, fuse=False, out_ave=True, num_concepts=64):
        super(ProDAS, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        clip_dim = self.clip_model.visual.output_dim
        self.vlam = VisionLanguageAttentiveMasker(clip_dim=clip_dim)
        
        num_scales = 4
        self.fsa_modules = nn.ModuleDict()
        self.text_proj_modules = nn.ModuleDict()
        for i in range(num_scales):
            dim = filters[i]
            self.text_proj_modules[f'scale_{i}'] = nn.Linear(clip_dim, dim)
            self.fsa_modules[f'scale_{i}'] = FactorizedSemanticAlignment(text_dim=dim, image_dim=dim, num_concepts=num_concepts)
        
        self.adaptive_weights = nn.Parameter(torch.ones(6) / 6)
        self.softmax = nn.Softmax(dim=0)
        
        self.block1_3 = ProDASBlock(num_channels, filters[0], dropout=dropout, up=False, down=True, fuse=fuse)
        self.block1_2 = ProDASBlock(filters[0], filters[0],  dropout=dropout, up=False, down=True, fuse=fuse)
        self.block1_1 = ProDASBlock(filters[0]*2, filters[0],  dropout=dropout, up=False, down=True, fuse=fuse)
        self.block10 = ProDASBlock(filters[0]*2, filters[0],  dropout=dropout, up=False, down=True, fuse=fuse)
        self.block11 = ProDASBlock(filters[0]*2, filters[0],  dropout=dropout, up=False, down=True, fuse=fuse)
        self.block12 = ProDASBlock(filters[0]*2, filters[0],  dropout=dropout, up=False, down=False, fuse=fuse)
        self.block13 = ProDASBlock(filters[0]*2, filters[0],  dropout=dropout, up=False, down=False, fuse=fuse)
        self.block2_2 = ProDASBlock(filters[1], filters[1],  dropout=dropout, up=True, down=True, fuse=fuse)
        self.block2_1 = ProDASBlock(filters[1]*2, filters[1],  dropout=dropout, up=True, down=True, fuse=fuse)
        self.block20 = ProDASBlock(filters[1]*3, filters[1],  dropout=dropout, up=True, down=True, fuse=fuse)
        self.block21 = ProDASBlock(filters[1]*3, filters[1],  dropout=dropout, up=True, down=False, fuse=fuse)
        self.block22 = ProDASBlock(filters[1]*3, filters[1],  dropout=dropout, up=True, down=False, fuse=fuse)
        self.block3_1 = ProDASBlock(filters[2], filters[2],  dropout=dropout, up=True, down=True, fuse=fuse)
        self.block30 = ProDASBlock(filters[2]*2, filters[2],  dropout=dropout, up=True, down=False, fuse=fuse)
        self.block31 = ProDASBlock(filters[2]*3, filters[2],  dropout=dropout, up=True, down=False, fuse=fuse)
        self.block40 = ProDASBlock(filters[3], filters[3], dropout=dropout, up=True, down=False, fuse=fuse)
        
        self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        
        self.mam = MorphologicalAdaptationModule(feature_channels=filters[0])
        
        self.apply(InitWeights_He)
    
    def count_parameters(self, trainable_only=True):
        def count_params(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                return sum(p.numel() for p in module.parameters())
        
        total = count_params(self)
        clip_params = count_params(self.clip_model)
        trainable = total - clip_params if trainable_only else sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_dict = {
            'total_params': total,
            'trainable_params': trainable,
            'clip_params': clip_params,
            'vlam_params': count_params(self.vlam),
            'fsa_params': sum(count_params(m) for m in self.fsa_modules.values()),
            'encoder_params': sum([
                count_params(self.block1_3), count_params(self.block1_2), count_params(self.block1_1),
                count_params(self.block2_2), count_params(self.block2_1), count_params(self.block3_1)
            ]),
            'decoder_params': sum([
                count_params(self.block10), count_params(self.block11), count_params(self.block12),
                count_params(self.block13), count_params(self.block20), count_params(self.block21),
                count_params(self.block22), count_params(self.block30), count_params(self.block31),
                count_params(self.block40)
            ]),
            'mam_params': count_params(self.mam)
        }
        
        return param_dict
    
    def print_parameters(self):
        params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        
        print("=" * 60)
        print("ProDAS Model Parameter Summary")
        print("=" * 60)
        print(f"Total Parameters:        {params['total_params']:,}")
        print(f"Trainable Parameters:    {trainable_params['trainable_params']:,}")
        print(f"Frozen Parameters:       {params['total_params'] - trainable_params['trainable_params']:,}")
        print("-" * 60)
        print(f"CLIP Model:              {params['clip_params']:,} (frozen)")
        print(f"VLAM Module:             {params['vlam_params']:,}")
        print(f"FSA Modules:             {params['fsa_params']:,}")
        print(f"Encoder Blocks:          {params['encoder_params']:,}")
        print(f"Decoder Blocks:          {params['decoder_params']:,}")
        print(f"MAM Module:              {params['mam_params']:,}")
        print("=" * 60)
        
        return params

    def _apply_fsa_modulation(self, x, text_features, scale_idx):
        fsa = self.fsa_modules[f'scale_{scale_idx}']
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        x_enhanced = fsa(x_flat, text_features)
        return x_enhanced.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, x, texts=None):
        device = x.device
        
        initial_mask = None
        projected_text_features = {}

        if texts is not None:
            with torch.no_grad():
                image_for_clip = F.interpolate(x, size=self.clip_model.visual.input_resolution, mode='bicubic', align_corners=False)
                
                img_x = self.clip_model.visual.conv1(image_for_clip.type(self.clip_model.dtype))
                img_x = img_x.reshape(img_x.shape[0], img_x.shape[1], -1).permute(0, 2, 1)
                img_x = torch.cat([self.clip_model.visual.class_embedding.to(img_x.dtype) + torch.zeros(img_x.shape[0], 1, img_x.shape[-1], dtype=img_x.dtype, device=img_x.device), img_x], dim=1)
                img_x = img_x + self.clip_model.visual.positional_embedding.to(img_x.dtype)
                img_x = self.clip_model.visual.ln_pre(img_x)
                img_x = img_x.permute(1, 0, 2)
                img_x = self.clip_model.visual.transformer(img_x)
                img_x = img_x.permute(1, 0, 2)
                image_patch_features = self.clip_model.visual.ln_post(img_x[:, 1:, :]) @ self.clip_model.visual.proj
                image_patch_features = image_patch_features.to(x.dtype)

                text_tokens = clip.tokenize(texts).to(device)
                text_x = self.clip_model.token_embedding(text_tokens).type(self.clip_model.dtype)
                text_x = text_x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
                text_x = text_x.permute(1, 0, 2)
                text_x = self.clip_model.transformer(text_x)
                text_x = text_x.permute(1, 0, 2)
                
                text_features_proj = self.clip_model.ln_final(text_x) @ self.clip_model.text_projection
                
                sentence_features = text_features_proj[torch.arange(text_features_proj.shape[0]), text_tokens.argmax(dim=-1)]
                
                text_features_proj = text_features_proj.to(x.dtype)
                sentence_features = sentence_features.to(x.dtype)

            initial_mask = self.vlam(image_patch_features, text_features_proj, sentence_features)
            
            for i in range(len(self.fsa_modules)):
                projected_text_features[f'scale_{i}'] = self.text_proj_modules[f'scale_{i}'](text_features_proj)
        
        x1_3, x_down1_3 = self.block1_3(x)
        
        if texts is not None:
            x1_3 = self._apply_fsa_modulation(x1_3, projected_text_features['scale_0'], 0)
        
        x1_2, x_down1_2 = self.block1_2(x1_3)
        
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        if texts is not None:
            x2_2 = self._apply_fsa_modulation(x2_2, projected_text_features['scale_1'], 1)

        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        if texts is not None:
            x3_1 = self._apply_fsa_modulation(x3_1, projected_text_features['scale_2'], 2)

        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        
        if texts is not None:
            x_down3_1 = self._apply_fsa_modulation(x_down3_1, projected_text_features['scale_3'], 3)

        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))

        out1 = self.final1(x1_1)
        out2 = self.final2(x10)
        out3 = self.final3(x11)
        out4 = self.final4(x12)
        out5 = self.final5(x13)
        
        if texts is not None:
            weights = self.softmax(self.adaptive_weights)
            
            if initial_mask.shape[2:] != out1.shape[2:]:
                position_mask = F.interpolate(initial_mask, size=out1.shape[2:], mode='bilinear', align_corners=False)
            else:
                position_mask = initial_mask
            
            refined_mask = self.mam(x13, position_mask)
            
            position_mask = refined_mask.to(device)
            weights = weights.to(device)
            
            base_output = (
                weights[0] * out1 +
                weights[1] * out2 +
                weights[2] * out3 +
                weights[3] * out4 +
                weights[4] * out5
            )
            
            fused_output = base_output + weights[5] * position_mask
            
            return fused_output, [out1, out2, out3, out4, out5, initial_mask, refined_mask]
        
        if self.out_ave:
            output = (out1 + out2 + out3 + out4 + out5) / 5
        else:
            output = out5
            
        return output, [out1, out2, out3, out4, out5]


if __name__ == '__main__':
    print("Testing ProDAS Model Parameter Count...")
    print()
    
    model_cfg = {
        'num_classes': 1,
        'num_channels': 3,
        'feature_scale': 2,
        'dropout': 0.2,
        'fuse': True,
        'out_ave': True,
        'num_concepts': 64
    }
    
    model = ProDAS(**model_cfg)
    
    model.print_parameters()
    
    print("\nTest completed!")
