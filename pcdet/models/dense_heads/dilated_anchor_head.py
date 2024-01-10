import numpy as np
import torch.nn as nn

from .anchor_head_single import AnchorHeadSingle


class DilatedAnchorHead(AnchorHeadSingle):
    def __init__(self, model_cfg, input_channels, num_class, class_names, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        grid_size = model_cfg.get('GRID_SIZE')
        grid_size = np.round(grid_size).astype(np.int64)
        super().__init__(
            model_cfg=model_cfg, input_channels=input_channels, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )