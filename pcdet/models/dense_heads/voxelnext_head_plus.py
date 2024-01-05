import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils, common_utils, box_utils, box_coder_utils
from ...utils.spconv_utils import replace_feature, spconv
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

import copy
from easydict import EasyDict


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features

        return ret_dict


class VoxelNeXtHeadPlus(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        try:
            self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
            self.voxel_size = torch.Tensor(voxel_size).cuda()
            
        except:
            self.point_cloud_range = torch.Tensor(self.model_cfg.POINT_CLOUD_RANGE).cuda()
            self.voxel_size = torch.Tensor(self.model_cfg.VOXEL_SIZE).cuda()

        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.gaussian_ratio = self.model_cfg.get('GAUSSIAN_RATIO', 1)
        self.gaussian_type = self.model_cfg.get('GAUSSIAN_TYPE', ['nearst', 'gt_center'])
        # The iou branch is only used for Waymo dataset
        self.iou_branch = self.model_cfg.get('IOU_BRANCH', False)
        if self.iou_branch:
            self.rectifier = self.model_cfg.get('RECTIFIER')
            nms_configs = self.model_cfg.POST_PROCESSING.NMS_CONFIG
            self.nms_configs = [EasyDict(NMS_TYPE=nms_configs.NMS_TYPE, 
                                    NMS_THRESH=nms_configs.NMS_THRESH[i],
                                    NMS_PRE_MAXSIZE=nms_configs.NMS_PRE_MAXSIZE[i],
                                    NMS_POST_MAXSIZE=nms_configs.NMS_POST_MAXSIZE[i]) for i in range(num_class)]

        self.double_flip = self.model_cfg.get('DOUBLE_FLIP', False)
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.get('SHARED_CONV_CHANNEL', 128),
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

        ##########################################
        ##**************************************##
        iassd_target_cfg = self.model_cfg.IASSD_TARGET_CONFIG
        self.iassd_box_coder = getattr(box_coder_utils, iassd_target_cfg.BOX_CODER)(
            **iassd_target_cfg.BOX_CODER_CONFIG
        )
        ##**************************************##
        ##########################################

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RegLossSparse())
        self.add_module('ins_loss_func', loss_utils.WeightedClassificationLoss())
        if self.iou_branch:
            self.add_module('crit_iou', loss_utils.IouLossSparse())
            self.add_module('crit_iou_reg', loss_utils.IouRegLossSparse())

    def assign_targets(self, gt_boxes, num_voxels, spatial_indices, spatial_shape):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    num_voxels=num_voxels[bs_idx], spatial_indices=spatial_indices[bs_idx], 
                    spatial_shape=spatial_shape, 
                    feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])

            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)

        return ret_dict

    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, num_voxels, spatial_indices, spatial_shape, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)

        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride

        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)  #

        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            distance = self.distance(spatial_indices, center[k])
            inds[k] = distance.argmin()
            mask[k] = 1

            if 'gt_center' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], distance, radius[k].item() * self.gaussian_ratio)

            if 'nearst' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], self.distance(spatial_indices, spatial_indices[inds[k]]), radius[k].item() * self.gaussian_ratio)

            ret_boxes[k, 0:2] = center[k] - spatial_indices[inds[k]][:2]
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_index = self.forward_ret_dict['batch_index']

        tb_dict = {}
        loss = 0
        batch_indices = self.forward_ret_dict['voxel_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['voxel_indices'][:, 1:]

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            if self.iou_branch:
                batch_box_preds = self._get_predicted_boxes(pred_dict, spatial_indices)
                pred_boxes_for_iou = batch_box_preds.detach()
                iou_loss = self.crit_iou(pred_dict['iou'], target_dicts['masks'][idx], target_dicts['inds'][idx],
                                            pred_boxes_for_iou, target_dicts['gt_boxes'][idx], batch_indices)

                iou_reg_loss = self.crit_iou_reg(batch_box_preds, target_dicts['masks'][idx], target_dicts['inds'][idx],
                                                    target_dicts['gt_boxes'][idx], batch_indices)
                iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight'] if 'iou_weight' in self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS else self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                iou_reg_loss = iou_reg_loss * iou_weight #self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

                loss += (hm_loss + loc_loss + iou_loss + iou_reg_loss)
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
                tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
            else:
                loss += hm_loss + loc_loss

        ######################
        ##******************##
        sa_loss_cls, tb_dict_0 = self.get_sa_ins_layer_loss()
        tb_dict.update(tb_dict_0)
        loss += sa_loss_cls
        ##******************##
        ######################

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def _get_predicted_boxes(self, pred_dict, spatial_indices):
        center = pred_dict['center']
        center_z = pred_dict['center_z']
        #dim = pred_dict['dim'].exp()
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat((box_part_list), dim=-1)
        return pred_box

    def rotate_class_specific_nms_iou(self, boxes, scores, iou_preds, labels, rectifier, nms_configs):
        """
        :param boxes: (N, 5) [x, y, z, l, w, h, theta]
        :param scores: (N)
        :param thresh:
        :return:
        """
        assert isinstance(rectifier, list)

        box_preds_list, scores_list, labels_list = [], [], []
        for cls in range(self.num_class):
            mask = labels == cls
            boxes_cls = boxes[mask]
            scores_cls = torch.pow(scores[mask], 1 - rectifier[cls]) * torch.pow(iou_preds[mask].squeeze(-1), rectifier[cls])
            labels_cls = labels[mask]

            selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=scores_cls, box_preds=boxes_cls, 
                                                        nms_config=nms_configs[cls], score_thresh=None)

            box_preds_list.append(boxes_cls[selected])
            scores_list.append(scores_cls[selected])
            labels_list.append(labels_cls[selected])

        return torch.cat(box_preds_list, dim=0), torch.cat(scores_list, dim=0), torch.cat(labels_list, dim=0)

    def merge_double_flip(self, pred_dict, batch_size, voxel_indices, spatial_shape):
        # spatial_shape (Z, Y, X)
        pred_dict['hm'] = pred_dict['hm'].sigmoid()
        pred_dict['dim'] = pred_dict['dim'].exp()

        batch_indices = voxel_indices[:, 0]
        spatial_indices = voxel_indices[:, 1:]

        pred_dict_ = {k: [] for k in pred_dict.keys()}
        counts = []
        spatial_indices_ = []
        for bs_idx in range(batch_size):
            spatial_indices_batch = []
            pred_dict_batch = {k: [] for k in pred_dict.keys()}
            for i in range(4):
                bs_indices = batch_indices == (bs_idx * 4 + i)
                if i in [1, 3]:
                    spatial_indices[bs_indices, 0] = spatial_shape[0] - spatial_indices[bs_indices, 0]
                if i in [2, 3]:
                    spatial_indices[bs_indices, 1] = spatial_shape[1] - spatial_indices[bs_indices, 1]

                if i == 1:
                    pred_dict['center'][bs_indices, 1] = - pred_dict['center'][bs_indices, 1]
                    pred_dict['rot'][bs_indices, 1] *= -1
                    pred_dict['vel'][bs_indices, 1] *= -1

                if i == 2:
                    pred_dict['center'][bs_indices, 0] = - pred_dict['center'][bs_indices, 0]
                    pred_dict['rot'][bs_indices, 0] *= -1
                    pred_dict['vel'][bs_indices, 0] *= -1

                if i == 3:
                    pred_dict['center'][bs_indices, 0] = - pred_dict['center'][bs_indices, 0]
                    pred_dict['center'][bs_indices, 1] = - pred_dict['center'][bs_indices, 1]

                    pred_dict['rot'][bs_indices, 1] *= -1
                    pred_dict['rot'][bs_indices, 0] *= -1

                    pred_dict['vel'][bs_indices] *= -1

                spatial_indices_batch.append(spatial_indices[bs_indices])

                for k in pred_dict.keys():
                    pred_dict_batch[k].append(pred_dict[k][bs_indices])

            spatial_indices_batch = torch.cat(spatial_indices_batch)

            spatial_indices_unique, _inv, count = torch.unique(spatial_indices_batch, dim=0, return_inverse=True,
                                                               return_counts=True)
            spatial_indices_.append(spatial_indices_unique)
            counts.append(count)
            for k in pred_dict.keys():
                pred_dict_batch[k] = torch.cat(pred_dict_batch[k])
                features_unique = pred_dict_batch[k].new_zeros(
                    (spatial_indices_unique.shape[0], pred_dict_batch[k].shape[1]))
                features_unique.index_add_(0, _inv, pred_dict_batch[k])
                pred_dict_[k].append(features_unique)

        for k in pred_dict.keys():
            pred_dict_[k] = torch.cat(pred_dict_[k])
        counts = torch.cat(counts).unsqueeze(-1).float()
        voxel_indices_ = torch.cat([torch.cat(
            [torch.full((indices.shape[0], 1), i, device=indices.device, dtype=indices.dtype), indices], dim=1
        ) for i, indices in enumerate(spatial_indices_)])

        batch_hm = pred_dict_['hm']
        batch_center = pred_dict_['center']
        batch_center_z = pred_dict_['center_z']
        batch_dim = pred_dict_['dim']
        batch_rot_cos = pred_dict_['rot'][:, 0].unsqueeze(dim=1)
        batch_rot_sin = pred_dict_['rot'][:, 1].unsqueeze(dim=1)
        batch_vel = pred_dict_['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

        batch_hm /= counts
        batch_center /= counts
        batch_center_z /= counts
        batch_dim /= counts
        batch_rot_cos /= counts
        batch_rot_sin /= counts

        if not batch_vel is None:
            batch_vel /= counts

        return batch_hm, batch_center, batch_center_z, batch_dim, batch_rot_cos, batch_rot_sin, batch_vel, None, voxel_indices_

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices, spatial_shape):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
            'pred_ious': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            if self.double_flip:
                batch_hm, batch_center, batch_center_z, batch_dim, batch_rot_cos, batch_rot_sin, batch_vel, batch_iou, voxel_indices_ = \
                self.merge_double_flip(pred_dict, batch_size, voxel_indices.clone(), spatial_shape)
            else:
                batch_hm = pred_dict['hm'].sigmoid()
                batch_center = pred_dict['center']
                batch_center_z = pred_dict['center_z']
                batch_dim = pred_dict['dim'].exp()
                batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
                batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
                batch_iou = (pred_dict['iou'] + 1) * 0.5 if self.iou_branch else None
                batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
                voxel_indices_ = voxel_indices

            final_pred_dicts = centernet_utils.decode_bbox_from_voxels_nuscenes(
                batch_size=batch_size, indices=voxel_indices_,
                obj=batch_hm, 
                rot_cos=batch_rot_cos,
                rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z,
                dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                #circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if not self.iou_branch:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])
                ret_dict[k]['pred_ious'].append(final_dict['pred_ious'])

        for k in range(batch_size):
            pred_boxes = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            pred_scores = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            pred_labels = torch.cat(ret_dict[k]['pred_labels'], dim=0)
            if self.iou_branch:
                pred_ious = torch.cat(ret_dict[k]['pred_ious'], dim=0)
                pred_boxes, pred_scores, pred_labels = self.rotate_class_specific_nms_iou(pred_boxes, pred_scores, pred_ious, pred_labels, self.rectifier, self.nms_configs)

            ret_dict[k]['pred_boxes'] = pred_boxes
            ret_dict[k]['pred_scores'] = pred_scores
            ret_dict[k]['pred_labels'] = pred_labels + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels
    
############################################################
##********************************************************##
    def assign_stack_targets_IASSD(self, points, gt_boxes, extend_gt_boxes=None, weighted_labels=False,
                             ret_box_labels=False, ret_offset_labels=True,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0,
                             use_query_assign=False, central_radii=2.0, use_ex_gt_assign=False, fg_pc_ignore=False,
                             binary_label=False):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        box_idxs_labels = points.new_zeros(points.shape[0]).long() 
        gt_boxes_of_fg_points = []
        gt_box_of_points = gt_boxes.new_zeros((points.shape[0], 8))

        for k in range(batch_size):            
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if use_query_assign: ##
                centers = gt_boxes[k:k + 1, :, 0:3]
                query_idxs_of_pts = roiaware_pool3d_utils.points_in_ball_query_gpu(
                    points_single.unsqueeze(dim=0), centers.contiguous(), central_radii
                    ).long().squeeze(dim=0) 
                query_fg_flag = (query_idxs_of_pts >= 0)
                if fg_pc_ignore:
                    fg_flag = query_fg_flag ^ box_fg_flag 
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = query_fg_flag
                    box_idxs_of_pts = query_idxs_of_pts
            elif use_ex_gt_assign: ##
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                extend_fg_flag = (extend_box_idxs_of_pts >= 0)
                
                extend_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[box_fg_flag] #instance points should keep unchanged

                if fg_pc_ignore:
                    fg_flag = extend_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = extend_fg_flag 
                    box_idxs_of_pts = extend_box_idxs_of_pts 
                                
            elif set_ignore_flag: 
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            elif use_ball_constraint: 
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag

            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 or binary_label else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            bg_flag = (point_cls_labels_single == 0) # except ignore_id
            # box_bg_flag
            fg_flag = fg_flag ^ (fg_flag & bg_flag)
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            box_idxs_labels[bs_mask] = box_idxs_of_pts
            gt_box_of_points[bs_mask] = gt_boxes[k][box_idxs_of_pts]

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.iassd_box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single


        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
            'box_idxs_labels': box_idxs_labels,
            'gt_box_of_points': gt_box_of_points,
        }
        return targets_dict

    def assign_targets_IASSD(self, input_dict):
        """
        Args:
            input_dict:
                batch_size: int
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                centers_origin: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_coords: List of point_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            target_dict:
            ...
        """
        target_cfg = self.model_cfg.IASSD_TARGET_CONFIG
        gt_boxes = input_dict['gt_boxes']
        if gt_boxes.shape[-1] == 10:   #nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        targets_dict_center = {}
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = input_dict['batch_size']      
        if target_cfg.get('EXTRA_WIDTH', False):  # multi class extension
            extend_gt = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=target_cfg.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
        else:
            extend_gt = gt_boxes

        extend_gt_boxes = box_utils.enlarge_box3d(
            extend_gt.view(-1, extend_gt.shape[-1]), extra_width=target_cfg.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        if target_cfg.get('INS_AWARE_ASSIGN', False):
            sa_ins_labels, sa_gt_box_of_fg_points, sa_xyz_coords, sa_gt_box_of_points, sa_box_idxs_labels = [],[],[],[],[]
            sa_ins_preds = input_dict['sa_ins_preds']
            for i in range(1, len(sa_ins_preds)): # valid when i = 1,2 for IA-SSD
                # if sa_ins_preds[i].__len__() == 0:
                #     continue
                sa_xyz = input_dict['encoder_coords'][i]
                if i == 1:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]  #[0.2, 0.2, 0.2]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=True, use_ex_gt_assign= False 
                    )
                if i >= 2:
                # if False:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=False, use_ex_gt_assign= True 
                    )

                sa_xyz_coords.append(sa_xyz)
                sa_ins_labels.append(sa_targets_dict['point_cls_labels'])
                sa_gt_box_of_fg_points.append(sa_targets_dict['gt_box_of_fg_points'])
                sa_gt_box_of_points.append(sa_targets_dict['gt_box_of_points'])
                sa_box_idxs_labels.append(sa_targets_dict['box_idxs_labels'])                
                
            targets_dict_center['sa_ins_labels'] = sa_ins_labels
            targets_dict_center['sa_gt_box_of_fg_points'] = sa_gt_box_of_fg_points
            targets_dict_center['sa_xyz_coords'] = sa_xyz_coords
            targets_dict_center['sa_gt_box_of_points'] = sa_gt_box_of_points
            targets_dict_center['sa_box_idxs_labels'] = sa_box_idxs_labels
        
        return targets_dict_center

    def generate_sa_center_ness_mask(self):
        sa_pos_mask = self.forward_ret_dict['iassd_target_dicts']['sa_ins_labels']
        sa_gt_boxes = self.forward_ret_dict['iassd_target_dicts']['sa_gt_box_of_fg_points']
        sa_xyz_coords = self.forward_ret_dict['iassd_target_dicts']['sa_xyz_coords']
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)):
            pos_mask = sa_pos_mask[i] > 0
            gt_boxes = sa_gt_boxes[i]
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:]
            xyz_coords = xyz_coords[pos_mask].clone().detach()
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3]
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

            template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
            margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
            distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
            distance[:, 1, :] = -1 * distance[:, 1, :]
            distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
            distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

            centerness = distance_min / distance_max
            centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
            centerness = torch.clamp(centerness, min=1e-6)
            centerness = torch.pow(centerness, 1/3)

            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
            centerness_mask[pos_mask] = centerness

            sa_centerness_mask.append(centerness_mask)
        return sa_centerness_mask

    
    def get_sa_ins_layer_loss(self, tb_dict=None):
        sa_ins_labels = self.forward_ret_dict['iassd_target_dicts']['sa_ins_labels']
        sa_ins_preds = self.forward_ret_dict['iassd_target_dicts']['sa_ins_preds']
        sa_centerness_mask = self.generate_sa_center_ness_mask()
        sa_ins_loss, ignore = 0, 0
        for i in range(len(sa_ins_labels)): # valid when i =1, 2
            if len(sa_ins_preds[i]) != 0:
                try:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, self.num_class)
                except:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, 1)

            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].view(-1)
            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            if ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i+1][0]):
                centerness_mask = sa_centerness_mask[i]
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

            point_loss_ins = self.ins_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()        
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_ins = point_loss_ins * loss_weights_dict.get('ins_aware_weight',[1]*len(sa_ins_labels))[i]

            sa_ins_loss += point_loss_ins
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({
                'sa%s_loss_ins' % str(i): point_loss_ins.item(),
                'sa%s_pos_num' % str(i): pos_normalizer.item()
            })

        sa_ins_loss = sa_ins_loss / (len(sa_ins_labels) - ignore)
        tb_dict.update({
                'sa_loss_ins': sa_ins_loss.item(),
            })
        return sa_ins_loss, tb_dict

##********************************************************##
############################################################


    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def forward(self, data_dict):
        x = data_dict['encoded_spconv_tensor']

        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        self.forward_ret_dict['batch_index'] = batch_index
        
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], num_voxels, spatial_indices, spatial_shape
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        ###################################
        ##*******************************##
        iassd_ret_dict = {
                    'ctr_offsets': data_dict['ctr_offsets'],
                    'centers': data_dict['centers'],
                    'centers_origin': data_dict['centers_origin'],
                    'sa_ins_preds': data_dict['sa_ins_preds'],
                    }
        
        if self.training:
            iassd_targets_dict = self.assign_targets_IASSD(data_dict)
            iassd_ret_dict.update(iassd_targets_dict)
            self.forward_ret_dict['iassd_target_dicts'] = iassd_ret_dict
        ##*******************************##
        ###################################

        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['voxel_indices'] = voxel_indices

        if not self.training or self.predict_boxes_when_training:
            if self.double_flip:
                data_dict['batch_size'] = data_dict['batch_size'] // 4
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], 
                pred_dicts, voxel_indices, spatial_shape
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
