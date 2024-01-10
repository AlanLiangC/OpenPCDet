from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from ..model_utils.swin_utils import FFN, DropPath, to_2tuple, trunc_normal_, trunc_normal_init, constant_init
from ..backbones_image.swin import WindowMSA
from ...utils.spconv_utils import replace_feature, spconv
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils


class HeightMerging(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 patch_norm=False,
                 feature_map_size=[176,200],
                 bias=False,) -> None:
        super().__init__()
        self.patch_norm = patch_norm
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(in_channels)
        self.liner = nn.Linear(in_channels, out_channels, bias=bias)
        self.H, self.W = feature_map_size

    def forward(self, window_features, window_indices, _return_dense=False, _return_sparse=False):
        '''
        window_features: B*nW N C
        window_indices: B*nW N 3
        '''
        # window 2 sparse
        BW, N, C = window_features.shape
        sp_features = window_features.reshape(-1, C) # [12800, 128]
        window_indices = window_indices.reshape(-1, window_indices.shape[-1])
        x_mask = (window_indices[:,1] >= 0) & (window_indices[:,1] < 176)
        y_mask = (window_indices[:,2] >= 0) & (window_indices[:,2] < 200)
        window_mask = x_mask & y_mask
        window_indices_value = window_indices[window_mask]
        new_window_features = window_features.new_zeros(BW*N, self.out_channels)

        batch_size = int(window_indices[:,0].max() + 1)
        indices_unique, _inv = torch.unique(window_indices, dim=0,return_inverse=True)
        features_unique = sp_features.new_zeros((indices_unique.shape[0], C))
        features_unique.index_add_(0, _inv, sp_features)

        x_mask = (indices_unique[:,1] >= 0) & (indices_unique[:,1] < 176)
        y_mask = (indices_unique[:,2] >= 0) & (indices_unique[:,2] < 200)
        val_mask = x_mask & y_mask
        if self.patch_norm:
            sp_features = self.norm(features_unique[val_mask])

        sp_features = self.liner(sp_features) # [7872, 128]
        indices_unique = indices_unique[val_mask][:,[0,2,1]]

        # sparse 2 window

        x_conv = spconv.SparseConvTensor(
            features=sp_features,
            indices=indices_unique.int(),
            spatial_shape=[self.W, self.H],
            batch_size=batch_size
        )

        if _return_sparse:
            return x_conv

        dense_feature = x_conv.dense() # [2, 200, 176, 128]
        if _return_dense:
            return dense_feature
        
        dense_feature = dense_feature.permute(0,2,3,1)
        dense_feature_flatten = dense_feature.flatten(0,2).unsqueeze(dim=0).permute(0,2,1).contiguous() # [70400, 128] -> [1, 128, 70400]
        window_indices_flatten = window_indices_value[:,0]*self.H*self.W + \
                                window_indices_value[:,2] * self.H + window_indices_value[:,1]
        window_features = pointnet2_utils.gather_operation(dense_feature_flatten, window_indices_flatten.unsqueeze(dim=0).int().contiguous())
        new_window_features[window_mask] = window_features.squeeze().permute(1,0)
        window_indices = window_indices.reshape(BW, -1, 3)
        new_window_features = new_window_features.reshape(BW, N, self.out_channels)
    
        return new_window_features, window_indices

class DilatedWindowMSA(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.)):
        super().__init__()

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,)
        self.drop = DropPath(dropout_layer['drop_prob'])

    def forward(self, query_windows):
        '''
        query_windows: nW*B, window_size*window_size, C
        '''
        attn_windows = self.w_msa(query_windows, mask=None)
        x = self.drop(attn_windows)
        return x

        
class Dilated2DBEVBackboneBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,):
        
        super(Dilated2DBEVBackboneBlock, self).__init__()
        
        self.with_cp = with_cp

        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = DilatedWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),)

        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,)

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class Dilated2DBEVBackboneBlockSequence(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 heightmerging=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Dilated2DBEVBackboneBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,)
            self.blocks.append(block)

        self.heightmerging = heightmerging
    
    def forward(self, x, window_indices):
        for block in self.blocks:
            x = block(x)
        
        if self.heightmerging:
            x, _new = self.heightmerging(x, window_indices)
        return x


class Dilated2DBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels) -> None:

        self.model_cfg = model_cfg
        feature_map_size = self.model_cfg.get('FEATURE_MAP_SIZE', (176,200))
        depths = self.model_cfg.get('DEPTHS', [2, 2])
        in_channels = input_channels
        embed_dims = input_channels
        num_heads = self.model_cfg.get('NUM_HEADS', [4,8])
        window_size = self.model_cfg.WINDOW_SIZE
        mlp_ratio = self.model_cfg.get('MLP_RATIO', 1)

        qkv_bias = self.model_cfg.get('QKV_BIAS', True)
        qk_scale = self.model_cfg.get('QK_SCALE', None)
        drop_rate = self.model_cfg.get('DROP_RATE', 0.)
        attn_drop_rate = self.model_cfg.get('ATTN_DROP_RATE', 0.)
        drop_path_rate = self.model_cfg.get('DROP_PATH_RATE', 0.2)
        patch_norm = self.model_cfg.get('PATCH_NORM', True)
        out_indices = self.model_cfg.get('OUT_INDICES', [0, 1])
        with_cp = self.model_cfg.get('WITH_CP', False)
        use_abs_pos_embed = self.model_cfg.get('USE_ABS_POS_EMBED', False)
        act_cfg=dict(type='GELU')
        norm_cfg=dict(type='LN')

        if isinstance(feature_map_size, int):
            feature_map_size = to_2tuple(feature_map_size)
        elif isinstance(feature_map_size, tuple):
            if len(feature_map_size) == 1:
                feature_map_size = to_2tuple(feature_map_size[0])
            assert len(feature_map_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(feature_map_size)}'
            
        super(Dilated2DBEVBackbone, self).__init__()
        self.init_cfg = None
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        if self.use_abs_pos_embed:
            patch_row = feature_map_size[0]
            patch_col = feature_map_size[1]
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))
            
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.init_heightmerging = HeightMerging(in_channels=in_channels,
                                          out_channels=in_channels,
                                          patch_norm=patch_norm)

        self.stages = nn.ModuleList()
        in_channels = embed_dims

        for i in range(num_layers):
            heightmerging = HeightMerging(in_channels=in_channels,
                                          out_channels=2*in_channels,
                                          patch_norm=patch_norm)
            
            stage = Dilated2DBEVBackboneBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                heightmerging=heightmerging,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp
            )
            self.stages.append(stage)
            if heightmerging:
                in_channels = heightmerging.out_channels

        self.num_features = [int(embed_dims * 2 * (i+1)) for i in range(num_layers)]
        self.num_bev_features = self.num_features[-1]
        self.final_heightmerging = HeightMerging(in_channels=self.num_bev_features,
                                          out_channels=int(self.num_bev_features),
                                          patch_norm=patch_norm)
        # Add a norm layer for each output

    def init_weights(self):
        if self.init_cfg is None:
            print(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)

    def forward(self, batch_dict):

        window_indices = batch_dict['window_indices'] # [B*nW, L, C]
        window_features = batch_dict['window_features'] # [B*nW, L, C]

        x, temp_indices = self.init_heightmerging(window_features, window_indices)
        assert temp_indices[0,16,2] == window_indices[0,16,2]

        if self.use_abs_pos_embed:
            x = window_features + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for i, stage in enumerate(self.stages):
            x = stage(x, window_indices)

        x = self.final_heightmerging(x, window_indices, _return_dense=True)
        batch_dict['spatial_features_2d'] = x #
        return batch_dict
