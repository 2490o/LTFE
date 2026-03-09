import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tF

from typing import Dict,List,Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer



class LTFE(nn.Module):
    def __init__(self, in_channels=1024, T_train=8, T_test=2):
        super().__init__()
        self.T_train = T_train
        self.T_test = T_test
        self.alpha_0 = 0.2
        self.sigma_0 = 1.0
        self.gamma = 1.2
        self.lambda_ = 0.2


        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=256, batch_first=True)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + 256, in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )


        self.ode_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, in_channels * 9)  # 对应 3x3 depthwise 卷积核
        )

        self.W_0 = nn.Parameter(torch.zeros(in_channels, 1, 3, 3))
        nn.init.dirac_(self.W_0)

    def apply_gaussian_blur(self, x, sigma):
        kernel_size = max(3, int(2 * math.ceil(2 * sigma) + 1))
        if kernel_size % 2 == 0: kernel_size += 1
        return tF.gaussian_blur(x, [kernel_size, kernel_size], [sigma, sigma])

    def forward(self, F_0):
        B, C, H, W = F_0.shape
        T_steps = self.T_train if self.training else self.T_test

        F_t = F_0
        F_seq = []

        for t in range(1, T_steps + 1):
            sigma_t = self.sigma_0 * (self.gamma ** t)
            alpha_t = self.alpha_0 * math.exp(-self.lambda_ * t)

            noise = torch.randn_like(F_0)
            blurred_F = self.apply_gaussian_blur(F_t, sigma_t)
            blurred_noise = self.apply_gaussian_blur(noise, sigma_t)

            F_t = blurred_F + alpha_t * blurred_noise
            F_seq.append(F_t)


        F_seq_pooled = [F.adaptive_avg_pool2d(f, 1).view(B, C) for f in F_seq]
        F_seq_tensor = torch.stack(F_seq_pooled, dim=1)  # (B, T, C)
        lstm_out, _ = self.lstm(F_seq_tensor)  # (B, T, 256)

        F_hat_seq = []

        for t in range(T_steps):
            h_t = lstm_out[:, t, :]  # (B, 256)
            h_t_expanded = h_t.view(B, 256, 1, 1).expand(-1, -1, H, W)
            H_t_fused = self.fusion_conv(torch.cat([h_t_expanded, F_seq[t]], dim=1))

            H_t_pooled = F.adaptive_avg_pool2d(H_t_fused, 1).view(B, C)
            tau = torch.norm(H_t_pooled, p=2, dim=1, keepdim=True)
            tau = tau / (tau.max() + 1e-8)  # (B, 1) 时间积分步长近似

            dW = self.ode_net(h_t).view(B, C, 1, 3, 3)
            # 根据ODE计算当前的权重: W_t = W_0 + tau * dW
            W_t = self.W_0.unsqueeze(0) + tau.view(B, 1, 1, 1, 1) * dW

            # 为了高效地执行不同样本(Batch)的动态卷积，将其转化为分组卷积
            F_0_reshaped = F_0.view(1, B * C, H, W)
            W_t_reshaped = W_t.view(B * C, 1, 3, 3)

            F_hat_t = F.conv2d(F_0_reshaped, W_t_reshaped, padding=1, groups=B * C)
            F_hat_t = F_hat_t.view(B, C, H, W) + F_0  # 残差连接
            F_hat_seq.append(F_hat_t)

        return F_seq, F_hat_seq


# =========================================================================

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)

        self.ltfe = LTFE(in_channels=1024, T_train=8, T_test=2)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        clip_images = [T.functional.normalize(ci.flip(0) / 255, mean, std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i for i in clip_images])
        return clip_images

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        features_orig = features.copy()  # 保存源域特征供对齐使用


        if "res4" in features:
            _, F_hat_seq = self.ltfe(features["res4"])
            features = {"res4": F_hat_seq[-1]}  # 替换为最终演化特征

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
        features_orig = features.copy()

        # 推理阶段执行 2 步 LTFE 特征演化
        if "res4" in features:
            _, F_hat_seq = self.ltfe(features["res4"])
            features = {"res4": F_hat_seq[-1]}

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
        features_orig = features.copy()

        if "res4" in features:
            _, F_hat_seq = self.ltfe(features["res4"])
            features = {"res4": F_hat_seq[-1]}

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

