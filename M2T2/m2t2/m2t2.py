# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
'''
Top-level M2T2 network.
'''
import torch
import torch.nn as nn

from m2t2.action_decoder import ActionDecoder, infer_placements
from m2t2.contact_decoder import ContactDecoder
from m2t2.criterion import SetCriterion, GraspCriterion, PlaceCriterion
from m2t2.matcher import HungarianMatcher
from m2t2.pointnet2 import PointNet2MSG, PointNet2MSGCls


class M2T2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        object_encoder: nn.Module = None,
        grasp_mlp: nn.Module = None,
        set_criterion: nn.Module = None,
        grasp_criterion: nn.Module = None,
        place_criterion: nn.Module = None
    ):
        super(M2T2, self).__init__()
        self.backbone = backbone
        self.object_encoder = object_encoder
        self.transformer = transformer
        self.grasp_mlp = grasp_mlp
        self.set_criterion = set_criterion
        self.grasp_criterion = grasp_criterion
        self.place_criterion = place_criterion

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['backbone'] = PointNet2MSG.from_config(cfg.scene_encoder)
        channels = args['backbone'].out_channels
        obj_channels = None
        if cfg.contact_decoder.num_place_queries > 0:
            args['object_encoder'] = PointNet2MSGCls.from_config(
                cfg.object_encoder
            )
            obj_channels = args['object_encoder'].out_channels
            args['place_criterion'] = PlaceCriterion.from_config(
                cfg.place_loss
            )
        args['transformer'] = ContactDecoder.from_config(
            cfg.contact_decoder, channels, obj_channels
        )
        if cfg.contact_decoder.num_grasp_queries > 0:
            args['grasp_mlp'] = ActionDecoder.from_config(
                cfg.action_decoder, args['transformer']
            )
            matcher = HungarianMatcher.from_config(cfg.matcher)
            args['set_criterion'] = SetCriterion.from_config(
                cfg.grasp_loss, matcher
            )
            args['grasp_criterion'] = GraspCriterion.from_config(
                cfg.grasp_loss
            )
        return cls(**args)

    def forward(self, data, cfg):
        scene_feat = self.backbone(data['inputs'])
        object_inputs = data['object_inputs']
        object_feat = {}
        if self.object_encoder is not None:
            object_feat = self.object_encoder(object_inputs)
        if 'task_is_place' in data:
            for key, val in object_feat['features'].items():
                object_feat['features'][key] = (
                    val * data['task_is_place'].view(
                        data['task_is_place'].shape[0], 1, 1
                    )
                )
        lang_tokens = data.get('lang_tokens')
        embedding, outputs = self.transformer(
            scene_feat, object_feat, lang_tokens
        )

        losses = {}
        if self.place_criterion is not None:
            losses, stats = self.place_criterion(outputs, data)
            outputs[-1].update(stats)

        if self.set_criterion is not None:
            set_losses, outputs = self.set_criterion(outputs, data)
            losses.update(set_losses)
        else:
            outputs = outputs[-1]

        if self.grasp_mlp is not None:
            mask_features = scene_feat['features'][
                self.transformer.mask_feature
            ]
            obj_embedding = [emb[idx] for emb, idx in zip(
                embedding['grasp'], outputs['matched_idx']
            )]
            confidence = [
                mask.sigmoid() for mask in outputs['matched_grasping_masks']
            ]
            grasp_outputs = self.grasp_mlp(
                data['points'], mask_features, confidence,
                cfg.mask_thresh, obj_embedding, data['grasping_masks']
            )
            outputs.update(grasp_outputs)
            contact_losses = self.grasp_criterion(outputs, data)
            losses.update(contact_losses)

        return outputs, losses

    def infer(self, data, cfg):
        scene_feat = self.backbone(data['inputs'])
        object_feat = self.object_encoder(data['object_inputs'])
        if 'task_is_place' in data:
            for key in object_feat['features']:
                object_feat['features'][key] = (
                    object_feat['features'][key] * data['task_is_place'].view(
                        data['task_is_place'].shape[0], 1, 1
                    )
                )
        lang_tokens = data.get('lang_tokens')
        embedding, outputs = self.transformer(
            scene_feat, object_feat, lang_tokens
        )
        outputs = outputs[-1]

        if 'place' in embedding and embedding['place'].shape[1] > 0:
            cam_pose = None if cfg.world_coord else data['cam_pose']
            placement_outputs = infer_placements(
                data['points'], outputs['placement_masks'],
                data['bottom_center'], data['ee_pose'],
                cam_pose, cfg.mask_thresh, cfg.placement_height
            )
            outputs.update(placement_outputs)
            outputs['placement_masks'] = (
                outputs['placement_masks'].sigmoid() > cfg.mask_thresh
            )

        if 'grasp' in embedding and embedding['grasp'].shape[1] > 0:
            masks = outputs['grasping_masks'].sigmoid() > cfg.mask_thresh
            mask_features = scene_feat['features'][
                self.transformer.mask_feature
            ]
            if 'objectness' in outputs:
                objectness = outputs['objectness'].sigmoid()
                object_ids = [
                    torch.where(
                        (score > cfg.object_thresh) & mask.sum(dim=1) > 0
                    )[0]
                    for score, mask in zip(objectness, masks)
                ]
                outputs['objectness'] = [
                    score[idx] for score, idx in zip(objectness, object_ids)
                ]
                confidence = [
                    logits.sigmoid()[idx]
                    for logits, idx in zip(outputs['grasping_masks'], object_ids)
                ]
                outputs['grasping_masks'] = [
                    mask[idx] for mask, idx in zip(masks, object_ids)
                ]
                obj_embedding = [emb[idx] for emb, idx in zip(
                    embedding['grasp'], object_ids
                )]
            else:
                obj_embedding = embedding['grasp']
                confidence = [
                    logits.sigmoid() for logits in outputs['grasping_masks']
                ]
            grasp_outputs = self.grasp_mlp(
                data['points'], mask_features, confidence,
                cfg.mask_thresh, obj_embedding
            )
            outputs.update(grasp_outputs)

        return outputs
