from pydoc import classname
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T

from detectron2.layers import ShapeSpec
from detectron2.data import MetadataCatalog

from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from .box_predictor import ClipFastRCNNOutputLayers


def select_foreground_proposals(
        proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        clsnames = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes").copy()

        for name in cfg.MODEL.RENAME:
            ind = clsnames.index(name[0])
            clsnames[ind] = name[1]

        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3)  ### copied
        self.box_predictor = ClipFastRCNNOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1),
                                                      clsnames)
        self.clip_im_predictor = self.box_predictor.cls_score
        self.device = cfg.MODEL.DEVICE

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            crops: Optional[List[Tuple]] = None,
            features_orig: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            loss_crop_im = None
            if crops is not None:
                crop_im = list()
                crop_boxes = list()
                keep = torch.ones(len(crops)).bool()

                for ind, x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))

                c = self._shared_roi_transform(
                    [features[f][keep] for f in self.in_features], crop_boxes)
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im, crops_features.mean(dim=[2, 3]))

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if loss_crop_im is not None:
                losses.update(loss_crop_im)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeadsAttn(ClipRes5ROIHeads):
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.fwdres5(x)

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            crops: Optional[List[Tuple]] = None,
            backbone=None,
            features_orig: Optional[Dict[str, torch.Tensor]] = None,
    ):
        del images
        self.fwdres5 = backbone.forward_res5

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            loss_crop_im = None
            if crops is not None:
                crop_im = list()
                crop_boxes = list()
                keep = torch.ones(len(crops)).bool()

                for ind, x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))

                crops_features = self._shared_roi_transform(
                    [features[f][keep] for f in self.in_features], crop_boxes)
                crops_features = backbone.attention_global_pool(crops_features)
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im, crops_features)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]


        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        attn_feat = backbone.attention_global_pool(box_features)

        if hasattr(attn_feat, 'shape') and len(attn_feat.shape) > 2:
            attn_feat_pooled = attn_feat.mean(dim=(2, 3))
        else:
            attn_feat_pooled = attn_feat

        if isinstance(box_features, list):
            box_features_pooled = box_features.mean(dim=(2, 3))
        else:
            box_features_pooled = box_features.mean(dim=(2, 3))

        predictions = self.box_predictor([attn_feat, box_features_pooled])

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)


            if features_orig is not None:
                with torch.no_grad():
                    box_features_orig = self._shared_roi_transform(
                        [features_orig[f] for f in self.in_features], proposal_boxes
                    )
                    attn_feat_orig = backbone.attention_global_pool(box_features_orig)
                    if hasattr(attn_feat_orig, 'shape') and len(attn_feat_orig.shape) > 2:
                        P_in = attn_feat_orig.mean(dim=(2, 3))
                    else:
                        P_in = attn_feat_orig

                P_hat = attn_feat_pooled

                labels = torch.cat([p.gt_classes for p in proposals], dim=0)
                valid_mask = (labels >= 0) & (labels != self.num_classes)

                if valid_mask.sum() > 0:
                    P_in_v = P_in[valid_mask]
                    P_hat_v = P_hat[valid_mask]
                    labels_v = labels[valid_mask]


                    L_intra = F.mse_loss(P_in_v, P_hat_v)

                    # 2. Inter-class Separability Loss
                    P_in_norm = F.normalize(P_in_v, p=2, dim=1)
                    P_hat_norm = F.normalize(P_hat_v, p=2, dim=1)
                    sim_matrix = torch.matmul(P_in_norm, P_hat_norm.t())  # Cosine similarity

                    sim_ii = torch.diag(sim_matrix)
                    num = torch.exp(sim_ii)


                    labels_i = labels_v.unsqueeze(1)
                    labels_j = labels_v.unsqueeze(0)
                    mask_diff_cls = (labels_i != labels_j).float()

                    den = (torch.exp(sim_matrix) * mask_diff_cls).sum(dim=1) + 1e-8

                    L_inter = -torch.log(num / (num + den)).mean()


                    loss_align = 1.0 * L_intra + 0.1 * L_inter
                    losses["loss_align"] = loss_align
            # =========================================================================

            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if loss_crop_im is not None:
                losses.update(loss_crop_im)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features_orig if features_orig is not None else features,
                                                           pred_instances)
            return pred_instances, {}