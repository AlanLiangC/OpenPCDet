from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import pointnet2_batch_cuda as pointnet2


class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative farthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.farthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = furthest_point_sample = FarthestPointSampling.apply

class FurthestPointSamplingWithDist(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, N) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        return_int = pointnet2.furthest_point_sampling_with_dist_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()

        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply

class SparseGroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (M, nsample) tensor containing the indicies of features to group with
        :param indices: (M, 2)
        :return:
            output: (M, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert indices.is_contiguous()

        M, nsample = idx.size()
        B, C, N = features.size()
        output = torch.cuda.FloatTensor(M, C, nsample)

        pointnet2.group_points_sparse_wrapper(B, C, N, M, nsample, features, indices, idx, output)

        ctx.save_for_backward(idx, indices, features)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, indices, features, = ctx.saved_tensors
        B, _, N = features.shape

        M, C, nsample = grad_out.size()
        grad_features = grad_out.new_zeros(B, C, N)

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_sparse_grad_wrapper(B, C, N, M, nsample, grad_out_data, indices, idx, grad_features)
        return grad_features, None, None

sparse_grouping_operation = SparseGroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz) # [8, 4096, 16]
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features

class SparseBallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, K, 3) centers of the ball query
        :param indices: (M, 2)
        :return:
            idx: (M, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert indices.is_contiguous()

        B, N, _ = xyz.size()
        K = new_xyz.size(1)
        M = indices.size(0)
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        pointnet2.ball_query_sparse_wrapper(B, N, M, K, radius, nsample, new_xyz, xyz, indices, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

sparse_ball_query = SparseBallQuery.apply

class BallQueryDilated(Function):

    @staticmethod
    def forward(ctx, max_radius: float, min_radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param max_radius: float, max radius of the balls
        :param min_radius: float, min radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_dilated_wrapper(B, N, npoint, max_radius, min_radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

ball_query_dilated = BallQueryDilated.apply

class SparseIndexingGet(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        :param feature: (B, C, N)
        :param indices: (M, 2)
        :return:
             output: (M, C)
        """
        assert feature.is_contiguous()
        assert indices.is_contiguous()

        B, C, N = feature.size()
        M, _ = indices.size()
        output = feature.new_empty(M, C)
        ctx.input_shape = feature.size()

        ctx.save_for_backward(indices)

        pointnet2.sparse_indexing_get_wrapper(B, C, N, M, feature, indices, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices,  = ctx.saved_tensors
        B, C, N = ctx.input_shape
        M, _ = indices.size()

        grad_feature = grad_output.new_empty(B, C, N)
        pointnet2.sparse_indexing_put_wrapper(B, C, N, M, grad_output, indices, grad_feature)
        return grad_feature, None

sparse_indexing_get = SparseIndexingGet.apply

class SparseIndexingPut(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, indices: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        :param feature: (M, C)
        :param indices: (M, 2)
        :param output: (B, C, N)
        :return:
             output: (B, C, N)
        """
        assert feature.is_contiguous()
        assert indices.is_contiguous()
        assert output.is_contiguous()

        B, C, N = output.size()
        M, _ = feature.size()

        ctx.save_for_backward(indices)

        pointnet2.sparse_indexing_put_wrapper(B, C, N, M, feature, indices, output)
        return output

    @staticmethod
    def backward(ctx, feature, grad_output):
        indices, = ctx.saved_tensors
        B, C, N = grad_output.shape 
        M, _ = indices.size()

        grad_feature = feature.new_empty(M, C)
        pointnet2.sparse_indexing_get_wrapper(B, C, N, M, grad_output, indices, grad_feature)
        return grad_feature, None, None

sparse_indexing_put = SparseIndexingPut.apply

class SparseIndexingReplace(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, indices: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        :param feature: (M, C)
        :param indices: (M, 2)
        :param output: (B, C, N)
        :return:
             output: (B, C, N)
        """
        assert feature.is_contiguous()
        assert indices.is_contiguous()
        assert output.is_contiguous()

        B, C, N = output.size()
        M, _ = feature.size()

        ctx.save_for_backward(indices)

        pointnet2.sparse_indexing_replace_wrapper(B, C, N, M, feature, indices, output)
        return output

    @staticmethod
    def backward(ctx, feature, grad_output):
        indices, = ctx.saved_tensors
        B, C, N = grad_output.shape 
        M, _ = indices.size()

        grad_feature = feature.new_empty(M, C)
        pointnet2.sparse_indexing_get_wrapper(B, C, N, M, grad_output, indices, grad_feature)
        return grad_feature, None, None

sparse_indexing_replace = SparseIndexingReplace.apply

class SparseQueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, indices: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, K, 3) centroids
        :param indices: (M, 2)
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (M, 3 + C, nsample)
        """
        idx = sparse_ball_query(self.radius, self.nsample, xyz, new_xyz, indices) # [8192, 16]
        xyz_trans = xyz.transpose(1, 2).contiguous()
        new_xyz_trans = new_xyz.transpose(1, 2).contiguous()
        grouped_xyz = sparse_grouping_operation(xyz_trans, idx, indices)  # (M, 3, nsample)
        with torch.no_grad():
            new_xyz_trans = sparse_indexing_get(new_xyz_trans, indices).unsqueeze(-1) 
        grouped_xyz -= new_xyz_trans 

        if features is not None:
            grouped_features = sparse_grouping_operation(features, idx, indices)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M, C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features



class QueryDilatedAndGroup(nn.Module):
    def __init__(self, radius_in: float, radius_out: float, nsample: int, use_xyz: bool = True):
        """
        :param radius_in: float, radius of inner ball
        :param radius_out: float, radius of outer ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius_in, self.radius_out, self.nsample, self.use_xyz = radius_in, radius_out, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
            idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
        """
        idx = ball_query_dilated(self.radius_in, self.radius_out, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features

class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
