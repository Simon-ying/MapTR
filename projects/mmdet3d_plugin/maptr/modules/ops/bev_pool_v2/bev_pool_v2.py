import bev_pool_v2_ext
import torch

def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """
    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_depth: (N_points, ),
        ranks_feat:  (N_points, ),
        ranks_bev:   (N_points, ),
        bev_feat_shape: (B, D_Z, D_Y, D_X, C)
        interval_starts: (N_pillar, )
        interval_lengths: (N_pillar, )
    Returns:
        x: bev feature in shape (B, C, Dz, Dy, Dx)
    """
    x = QuickCumsumCuda.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths)      # (B, Dz, Dy, Dx, C)
    x = x.permute(0, 4, 1, 2, 3).contiguous()        # (B, C, Dz, Dy, Dx)
    return x
 
 
class QuickCumsumCuda(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.
    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()     # (N_points, ),
        depth = depth.contiguous().float()  # (B, N, D, fH, fW)
        feat = feat.contiguous().float()    # (B, N, fH, fW, C)
        ranks_depth = ranks_depth.contiguous().int()    # (N_points, ),
        ranks_feat = ranks_feat.contiguous().int()      # (N_points, ),
        interval_lengths = interval_lengths.contiguous().int()  # (N_pillar, )
        interval_starts = interval_starts.contiguous().int()    # (N_pillar, )
 
        out = feat.new_zeros(bev_feat_shape)    # (B, D_Z, D_Y, D_X, C)
 
        bev_pool_v2_ext.bev_pool_v2_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )
 
        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out