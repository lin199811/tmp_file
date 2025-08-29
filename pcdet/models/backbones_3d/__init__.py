from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x, LVoxelBackBone8x, LVoxelResBackBone8x, VoxelWideResBackBone8x, VoxelWideResBackBone_L8x
from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
from .spconv_backbone_unibn import VoxelBackBone8x_UniBN, VoxelResBackBone8x_UniBN
from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_unet import UNetV2
from .IASSD_backbone import IASSD_Backbone
from .dsvt import DSVT
from .spconv2d_backbone_pillar import PillarRes18BackBone_one_stride
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D

from .spconv_backbone_sed import HEDNet
from .hednet import SparseHEDNet, SparseHEDNet2D
from .lion_backbone_one_stride import LION3DBackboneOneStride, LION3DBackboneOneStride_Sparse

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'LVoxelBackBone8x': LVoxelBackBone8x,
    'LVoxelResBackBone8x': LVoxelResBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelWideResBackBone8x': VoxelWideResBackBone8x,
    'VoxelWideResBackBone_L8x': VoxelWideResBackBone_L8x,
    # Dataset-specific Norm Layer
    'VoxelBackBone8x_UniBN':VoxelBackBone8x_UniBN,
    'VoxelResBackBone8x_UniBN':VoxelResBackBone8x_UniBN,
    'IASSD_Backbone': IASSD_Backbone,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
    'PillarRes18BackBone_one_stride': PillarRes18BackBone_one_stride,
    'VoxelResBackBone8xVoxelNeXt': VoxelResBackBone8xVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXt2D': VoxelResBackBone8xVoxelNeXt2D,
    'PillarBackBone8x': PillarBackBone8x,
    'PillarRes18BackBone8x': PillarRes18BackBone8x,
    'DSVT': DSVT,
    'HEDNet': HEDNet,
    'SparseHEDNet': SparseHEDNet,
    'SparseHEDNet2D': SparseHEDNet2D,
    'LION3DBackboneOneStride': LION3DBackboneOneStride,
    'LION3DBackboneOneStride_Sparse': LION3DBackboneOneStride_Sparse,
}
