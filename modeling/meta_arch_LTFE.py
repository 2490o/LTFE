import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tF

from typing import Dict, List, Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer


# =========================================================================
# 新增模块：Liquid Temporal Feature Evolution (LTFE)
# =========================================================================
class LTFE(nn.Module):
    def __init__(self, in_channels=1024, T_train=8, T_test=2):
        super().__init__()
        self.T_train = T_train
        self.T_test = T_test
        self.alpha_0 = 0.2
        self.sigma_0 = 1.0
        self.gamma = 1.2
        self.lambda_ = 0.2

        # TDM: 轻量化 LSTM 建模
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=256, batch_first=True)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + 256, in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # LPE: ODE 近似网络，用于生成动态 Depthwise 卷积残差参数
        self.ode_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, in_channels * 9),  # 对应 3x3 depthwise 卷积核
            nn.Tanh()  # [FIX 1] 强制限制输出在 [-1, 1] 之间，防止动态卷积权重(dW)爆炸
        )

        # 初始化恒等映射卷积核
        self.W_0 = nn.Parameter(torch.zeros(in_channels, 1, 3, 3))
        nn.init.dirac_(self.W_0)

        # [FIX 2] 新增归一化层：用于稳定动态卷积后的特征方差
        self.norm = nn.GroupNorm(32, in_channels)
        # Gaussian blur kernel cap: huge kernels on large res4 maps can stress memory / cuda path.
        self._gaussian_kernel_max = 21

    def apply_gaussian_blur(self, x, sigma):
        kernel_size = max(3, int(2 * math.ceil(2 * sigma) + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, self._gaussian_kernel_max)
        sigma = float(min(sigma, 0.5 * (kernel_size - 1)))
        return tF.gaussian_blur(x, [kernel_size, kernel_size], [sigma, sigma])

    @staticmethod
    def _depthwise_conv3x3_per_sample(x_bc_hw, w_b_c_3_3):
        """
        x: (B, C, H, W), w: (B, C, 1, 3, 3) — avoid cudnn instability from groups=B*C fused conv.
        """
        B, C, H, W = x_bc_hw.shape
        outs = []
        for bi in range(B):
            outs.append(
                F.conv2d(x_bc_hw[bi : bi + 1], w_b_c_3_3[bi], padding=1, groups=C)
            )
        return torch.cat(outs, dim=0)

    def forward(self, F_0):
        in_dtype = F_0.dtype
        # FP32 inside LTFE avoids half-precision edge cases on dynamic depthwise / LSTM / blur.
        F_0 = F_0.float()

        B, C, H, W = F_0.shape
        T_steps = self.T_train if self.training else self.T_test

        F_t = F_0
        F_seq = []
        # 1. PTFE: 渐进式时序特征演化
        for t in range(1, T_steps + 1):
            sigma_t = self.sigma_0 * (self.gamma ** t)
            alpha_t = self.alpha_0 * math.exp(-self.lambda_ * t)

            noise = torch.randn_like(F_0)
            blurred_F = self.apply_gaussian_blur(F_t, sigma_t)
            blurred_noise = self.apply_gaussian_blur(noise, sigma_t)

            F_t = blurred_F + alpha_t * blurred_noise
            F_seq.append(F_t)

        # 2. TDM: 空间池化后通过 LSTM 提取依赖，降低计算复杂度
        F_seq_pooled = [F.adaptive_avg_pool2d(f, 1).view(B, C) for f in F_seq]
        F_seq_tensor = torch.stack(F_seq_pooled, dim=1)  # (B, T, C)
        lstm_out, _ = self.lstm(F_seq_tensor)  # (B, T, 256)

        F_hat_seq = []
        # 3. LPE: 动态参数生成与特征调整
        for t in range(T_steps):
            h_t = lstm_out[:, t, :]  # (B, 256)
            h_t_expanded = h_t.view(B, 256, 1, 1).expand(-1, -1, H, W)
            H_t_fused = self.fusion_conv(torch.cat([h_t_expanded, F_seq[t]], dim=1))

            H_t_pooled = F.adaptive_avg_pool2d(H_t_fused, 1).view(B, C)
            tau = torch.norm(H_t_pooled, p=2, dim=1, keepdim=True)

            # [FIX 3] 安全的归一化：限制 tau_max 防止除以 0 或被 Inf 破坏引发 NaN
            tau_max = torch.clamp(tau.max(), min=1e-8, max=1e4)
            tau = tau / tau_max  # (B, 1) 时间积分步长近似

            dW = self.ode_net(h_t).view(B, C, 1, 3, 3)
            # 根据ODE计算当前的权重: W_t = W_0 + tau * dW
            W_t = self.W_0.unsqueeze(0) + tau.view(B, 1, 1, 1, 1) * dW

            F_hat_t = self._depthwise_conv3x3_per_sample(F_0, W_t)

            # [FIX 2 续] 动态卷积后先进行归一化，再加入原始特征残差，防止方差失控
            F_hat_t = self.norm(F_hat_t) + F_0
            F_hat_seq.append(F_hat_t)

        if in_dtype != torch.float32:
            F_seq = [t.to(in_dtype) for t in F_seq]
            F_hat_seq = [t.to(in_dtype) for t in F_hat_seq]

        return F_seq, F_hat_seq


# =========================================================================

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
        # 初始化 LTFE (ResNet 第4层提取的通道数为 1024)
        self.ltfe = LTFE(in_channels=1024, T_train=8, T_test=2)
        # 保守默认：避免 LTFE 直接覆盖原始语义导致源域性能下降
        self.ltfe_mix_ratio = float(getattr(cfg.MODEL, "LTFE_MIX_RATIO", 0.25))
        self.ltfe_min_cos = float(getattr(cfg.MODEL, "LTFE_MIN_COS", 0.30))

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        clip_images = [T.functional.normalize(ci.flip(0) / 255, mean, std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i for i in clip_images])
        return clip_images

    def _evolve_features(self, features: Dict[str, torch.Tensor]):
        """
        Keep proposals on original features and apply LTFE conservatively on ROI features.
        """
        features_orig = features.copy()
        features_evolved = features.copy()
        if "res4" not in features:
            return features_evolved, features_orig

        res4_orig = features["res4"]
        _, F_hat_seq = self.ltfe(res4_orig)
        res4_evolved = F_hat_seq[-1]

        # Adaptive mixing: if evolved feature drifts too far, reduce LTFE contribution.
        with torch.no_grad():
            v_orig = F.adaptive_avg_pool2d(res4_orig, 1).flatten(1)
            v_evo = F.adaptive_avg_pool2d(res4_evolved, 1).flatten(1)
            cos_sim = F.cosine_similarity(v_orig, v_evo, dim=1, eps=1e-6).mean()
            drift_factor = ((cos_sim - self.ltfe_min_cos) / (1.0 - self.ltfe_min_cos)).clamp(0.0, 1.0)
            mix = float(self.ltfe_mix_ratio) * float(drift_factor.item())

        features_evolved["res4"] = (1.0 - mix) * res4_orig + mix * res4_evolved
        return features_evolved, features_orig

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        features, features_orig = self._evolve_features(features)

        if self.proposal_generator is not None:
            # 论文中：使用原始特征 F_0 获取 Proposal
            logits, proposals, proposal_losses = self.proposal_generator(images, features_orig, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            # 传递演化特征 features 以及原始特征 features_orig 给 ROI 用于计算 TFAM 对齐
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone,
                                                features_orig)
        except Exception as e:
            print("ROI head exception:", e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None,
                                                features_orig=features_orig)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def generate_colors(self, N):
        import colorsys
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
        perm = np.arange(7)
        colors = [colors[idx] for idx in perm]
        return colors

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features, features_orig = self._evolve_features(features)

        if detected_instances is None:
            if self.proposal_generator is not None:
                logits, proposals, _ = self.proposal_generator(images, features_orig, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            try:
                results, _ = self.roi_heads(images, features, proposals, None, None, self.backbone, features_orig)
            except Exception as e:
                results, _ = self.roi_heads(images, features, proposals, None, None, features_orig=features_orig)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            return allresults
        else:
            return results


@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneWithOffsetGenTrainable(ClipRCNNWithClipBackbone):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        features, features_orig = self._evolve_features(features)

        if self.proposal_generator is not None:
            logits, proposals, proposal_losses = self.proposal_generator(images, features_orig, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone,
                                                features_orig)
        except Exception as e:
            print("ROI head exception:", e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None,
                                                features_orig=features_orig)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses