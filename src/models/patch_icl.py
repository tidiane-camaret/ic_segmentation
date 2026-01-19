"""
Patch-ICL Architecture
N levels that process the original inputs at progressively finer resolution levels

parameters (config.yaml -> model_params -> patch_icl)
model_params:
  patch_icl:
    res_levels: resolution for each level (length in voxels)
    patch_size: size of sampled patches for each level (length in voxels)
    num_train_patches: nb of sampled patches for each level
    num_valid_patches: 
    sampling_temperature
    backbone: backbone for all levels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.local import (
    LocalDino,

)
class PatchICL(nn.Module):
    """
    PatchICL segmentation model.

    Global branch: Uses GT mask (oracle) to produce coarse prediction.
    Patch selection: Selects K patches based on coarse prediction.
    Local branch: Shallow transformer refines selected patches.
    """

    def __init__(self, config: dict, context_size: int = 0):
        """
        Args:

        """
        super().__init__()


class PatchICL_Level(nn.Module):
    """
    ICL level
    
    """
"""
Level i
- inputs
    - I_t, M_t, I_c, M_c : target/context image masks at original resolution
    - M_t_prev_pred, M_c_prev_pred : target/context predicted masks at previous level
- parameters
    - context_size : nb of context images
    - res_i : resolution level (length in voxels)
    - patch_size : size of sampled patches (length in voxels)
    - nb_patches : number of sampled patches in target and context
- Logic
    1. Downsize I_t, M_t, I_c, M_c at resolution res_i
    2. Sample K patches in I_t and I_c
        - weighted by M_t_prev_pred and M_c_prev_pred
        - should sample more at boundaries
        - also includes rotations/transforms
        - differentiable sampling
        - Can sample only from patches with pred > 0.5
    3. Patch level prediction
        - predict mask for K*(1+context_size) patches using backbone
        - Can add a resolution embedding/token
        - Can use pre-computed features (e.g. Dino)
        - during training : random patch masking
        - compute patch_loss, patch_dice
    4.  Aggregation
        - Aggregate target and context patches in res_i masks
        - Can also include M_t_prev_pred and M_c_prev_pred
        - compute aggr_loss, aggr_dice
- outputs
    - M_t_prev_pred, M_c_prev_pred : target/context predicted masks
"""