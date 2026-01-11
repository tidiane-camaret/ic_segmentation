"""
Extended LightningModel with foreground-based context patch sampling.

This module provides a modified version of the Medverse LightningModel that samples
context patches from foreground locations rather than from the same spatial location
as the target patch during sliding window inference.
"""

import random
import torch
from monai.inferers.utils import sliding_window_inference
from monai.utils import BlendMode, PytorchPadMode

import sys
sys.path.insert(0, '/software/notebooks/camaret/repos/Medverse')
from medverse.lightning_model import LightningModel


class LightningModelForegroundSampling(LightningModel):
    """
    Extended LightningModel with foreground-based context patch sampling.

    Inherits from the original Medverse LightningModel and overrides only the
    sliding window inference method to sample context patches from foreground
    locations instead of the same spatial location as the target.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.enable_inspection = False
        self.inspection_data = None

    def enable_patch_inspection(self):
        """Enable detailed patch inspection during inference."""
        self.enable_inspection = True
        self.inspection_data = {
            'levels': [],  # List of levels, each containing patches
        }

    def disable_patch_inspection(self):
        """Disable patch inspection."""
        self.enable_inspection = False
        self.inspection_data = None

    def get_inspection_data(self):
        """Return collected inspection data."""
        return self.inspection_data

    def save_inspection_data_to_nifti(self, save_dir, case_id, label_id):
        """
        Save inspection data to NIfTI files with flat hierarchy.

        Args:
            save_dir: Directory to save files
            case_id: Case identifier (e.g., 's0032')
            label_id: Label identifier (e.g., 'heart')

        Files saved as: {save_dir}/{case_id}/{label_id}_level_{L}_patch_{P}_{type}.nii.gz
        """
        import nibabel as nib
        from pathlib import Path
        import numpy as np
        import json

        if self.inspection_data is None:
            print("No inspection data to save")
            return

        # Flat hierarchy: one directory per case
        case_dir = Path(save_dir) / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Collect metadata for all levels
        all_metadata = []

        # Save each level
        for level_data in self.inspection_data['levels']:
            level_num = level_data['level']

            # Store level metadata
            all_metadata.append({
                'level': level_num,
                'spatial_shape': list(level_data['spatial_shape']) if isinstance(level_data['spatial_shape'], tuple) else level_data['spatial_shape'],
                'roi_size': list(level_data['roi_size']) if isinstance(level_data['roi_size'], tuple) else level_data['roi_size'],
                'num_patches': len(level_data['patches'])
            })

            # Save each patch
            for patch_data in level_data['patches']:
                patch_id = patch_data['patch_id']
                prefix = f"{label_id}_level_{level_num}_patch_{patch_id:04d}"

                # Save target image
                target_in = patch_data['target_in'].numpy()  # [C, D, H, W]
                target_nifti = nib.Nifti1Image(target_in[0], affine=np.eye(4))
                nib.save(target_nifti, case_dir / f'{prefix}_target_in.nii.gz')

                # Save prediction
                prediction = patch_data['prediction'].numpy()  # [C, D, H, W]
                pred_nifti = nib.Nifti1Image(prediction[0], affine=np.eye(4))
                nib.save(pred_nifti, case_dir / f'{prefix}_prediction.nii.gz')

                # Save context data if available
                if 'context_in' in patch_data:
                    context_in = patch_data['context_in'].numpy()  # [L, C, D, H, W]
                    context_out = patch_data['context_out'].numpy()  # [L, C, D, H, W]
                    sampled_coords = patch_data['sampled_context_coords']

                    # Save each context example
                    for ctx_idx in range(context_in.shape[0]):
                        ctx_img = nib.Nifti1Image(context_in[ctx_idx, 0], affine=np.eye(4))
                        ctx_mask = nib.Nifti1Image(context_out[ctx_idx, 0], affine=np.eye(4))

                        nib.save(ctx_img, case_dir / f'{prefix}_context_{ctx_idx}_img.nii.gz')
                        nib.save(ctx_mask, case_dir / f'{prefix}_context_{ctx_idx}_mask.nii.gz')

                    # Save context coordinates
                    coords_dict = {
                        f'context_{i}': {'d': int(coords[0]), 'h': int(coords[1]), 'w': int(coords[2])}
                        for i, coords in enumerate(sampled_coords)
                    }
                    with open(case_dir / f'{prefix}_context_coordinates.json', 'w') as f:
                        json.dump(coords_dict, f, indent=2)

        # Save overall metadata
        with open(case_dir / f'{label_id}_metadata.json', 'w') as f:
            json.dump({'levels': all_metadata}, f, indent=2)

        print(f"Saved inspection data to {case_dir}")

    def _compute_foreground_patch_centers(self,
                                          context_out,
                                          roi_size,
                                          overlap,
                                          min_foreground_ratio=0.01):
        """
        Compute valid patch centers where context_out has foreground.

        Args:
            context_out: [C, D, H, W] - Single context segmentation mask
            roi_size: (D_roi, H_roi, W_roi) - Patch size
            overlap: Overlap ratio for patch sampling
            min_foreground_ratio: Minimum ratio of foreground voxels in patch (default 0.01 = 1%)

        Returns:
            List of (d, h, w) patch center coordinates with foreground
        """
        C, D, H, W = context_out.shape
        D_roi, H_roi, W_roi = roi_size

        # Calculate stride from overlap
        stride_d = max(1, int(D_roi * (1 - overlap)))
        stride_h = max(1, int(H_roi * (1 - overlap)))
        stride_w = max(1, int(W_roi * (1 - overlap)))

        foreground_centers = []
        threshold = min_foreground_ratio * (D_roi * H_roi * W_roi)

        # Iterate over all possible patch centers
        for d in range(0, D - D_roi + 1, stride_d):
            for h in range(0, H - H_roi + 1, stride_h):
                for w in range(0, W - W_roi + 1, stride_w):
                    # Extract patch
                    patch = context_out[:, d:d+D_roi, h:h+H_roi, w:w+W_roi]

                    # Check if patch has enough foreground
                    if patch.sum() >= threshold:
                        # Store patch top-left corner
                        foreground_centers.append((d, h, w))

        # If no foreground patches found, fall back to all possible centers
        if len(foreground_centers) == 0:
            for d in range(0, D - D_roi + 1, stride_d):
                for h in range(0, H - H_roi + 1, stride_h):
                    for w in range(0, W - W_roi + 1, stride_w):
                        foreground_centers.append((d, h, w))

        return foreground_centers

    def _sample_context_patches(self,
                                context_in,
                                context_out,
                                roi_size,
                                patch_centers,
                                num_samples=1):
        """
        Sample patches from context images at specified centers.

        Args:
            context_in: [L, C, D, H, W] - Context images
            context_out: [L, C, D, H, W] - Context masks
            roi_size: (D_roi, H_roi, W_roi) - Patch size
            patch_centers: List of lists of (d, h, w) centers for each context
            num_samples: Number of patches to sample from each context

        Returns:
            Tuple of (sampled_context_in, sampled_context_out)
            Each has shape [L, C, D_roi, H_roi, W_roi]
        """
        L, C, D, H, W = context_in.shape
        D_roi, H_roi, W_roi = roi_size

        sampled_in = []
        sampled_out = []

        for l_idx in range(L):
            # Randomly sample a patch center from available foreground centers
            if len(patch_centers[l_idx]) > 0:
                center = random.choice(patch_centers[l_idx])
            else:
                # Fallback to random center if no foreground found
                center = (
                    random.randint(0, max(0, D - D_roi)),
                    random.randint(0, max(0, H - H_roi)),
                    random.randint(0, max(0, W - W_roi))
                )

            d, h, w = center

            # Extract patches
            patch_in = context_in[l_idx:l_idx+1, :, d:d+D_roi, h:h+H_roi, w:w+W_roi]
            patch_out = context_out[l_idx:l_idx+1, :, d:d+D_roi, h:h+H_roi, w:w+W_roi]

            sampled_in.append(patch_in)
            sampled_out.append(patch_out)

        # Stack along L dimension
        sampled_in = torch.cat(sampled_in, dim=0)  # [L, C, D_roi, H_roi, W_roi]
        sampled_out = torch.cat(sampled_out, dim=0)  # [L, C, D_roi, H_roi, W_roi]

        return sampled_in, sampled_out

    def _sliding_window_autoregressive_step(self,
                                            current_target_in,
                                            current_context_in,
                                            current_context_out,
                                            image_context_in_prev,
                                            roi_size,
                                            sw_batch_size,
                                            overlap,
                                            mode,
                                            forward_l_param,
                                            current_level=None
                                            ):
        """
        Override of parent method with foreground-based context patch sampling.

        Helper function to perform a single step of sliding window inference for the autoregressive process.

        This function prepares inputs by stacking them, defines a predictor for MONAI's sliding_window_inference,
        and then invokes the sliding window inference.

        MODIFIED: Context patches are now sampled from foreground locations instead of same spatial location.

        Args:
            current_target_in (torch.Tensor): The target input for the current resolution level.
                                              Shape: [B, C_t, D_iter, H_iter, W_iter]
            current_context_in (torch.Tensor, optional): The semantic context input for the current resolution level.
                                                       Shape: [B, L_c, C_c, D_iter, H_iter, W_iter]
            current_context_out (torch.Tensor, optional): The semantic context output for the current resolution level.
                                                        Shape: [B, L_co, C_co, D_iter, H_iter, W_iter]
            image_context_in_prev (torch.Tensor, optional): The image context (output from the previous autoregressive level).
                                                          Shape: [B, 1, C_ic, D_iter, H_iter, W_iter]
            roi_size (tuple or list): The spatial size of the ROI for sliding window.
            sw_batch_size (int): The batch size for sliding window inference.
            overlap (float): The overlap ratio for sliding window.
            mode (BlendMode): The blending mode for overlapping windows (e.g., BlendMode.GAUSSIAN).
            forward_l_param (int): The 'l' parameter to be passed to the model's forward method.

        Returns:
            torch.Tensor: The prediction output from the sliding window inference for the current level.
        """
        if hasattr(self, 'concat_global_crop') and self.concat_global_crop:
            self.current_context_in_resized = torch.nn.functional.interpolate(current_context_in.squeeze(2), size=(128, 128, 128), mode='trilinear', align_corners=False).unsqueeze(2)
            self.current_context_out_resized = torch.nn.functional.interpolate(current_context_out.squeeze(2), size=(128, 128, 128), mode='trilinear', align_corners=False).unsqueeze(2)


        B_main = current_target_in.shape[0]  # Batch size for the main input (target_in)
        spatial_shape_iter = current_target_in.shape[2:] # Spatial dimensions for the current iteration

        # Initialize level inspection data if enabled
        if self.enable_inspection and current_level is not None:
            level_data = {
                'level': current_level,
                'spatial_shape': spatial_shape_iter,
                'roi_size': roi_size,
                'patches': []
            }
        else:
            level_data = None

        # Compute foreground patch centers for context sampling (NEW)
        foreground_patch_centers = None
        if current_context_out is not None and B_main == 1:
            L_c = current_context_out.shape[1]
            foreground_patch_centers = []
            for l_idx in range(L_c):
                ctx_out_single = current_context_out[0, l_idx]
                centers = self._compute_foreground_patch_centers(
                    ctx_out_single,
                    roi_size,
                    overlap,
                    min_foreground_ratio=0.01
                )
                foreground_patch_centers.append(centers)

        inputs_to_stack = []
        # Metadata to help reconstruct original tensor shapes and types from the stacked tensor within the predictor
        channel_meta = {
            "C_target": 0,      # Number of channels in target_in
            "original_L_c": 0,  # Original L dimension of context_in (number of context samples)
            "C_c": 0,           # Number of channels per context_in sample
            "original_L_co": 0, # Original L dimension of context_out
            "C_co": 0,          # Number of channels per context_out sample
            "C_ic": 0           # Number of channels in image_context_in_prev
        }
        flat_channel_counts = [] # Stores the number of channels for each input type *after* potential reshaping (e.g. L*C for contexts)

        # Stack target_in
        inputs_to_stack.append(current_target_in.to(self.device))
        channel_meta["C_target"] = current_target_in.shape[1]
        flat_channel_counts.append(channel_meta["C_target"])

        # Don't stack context - we'll sample it inside predictor (MODIFIED)
        if current_context_in is not None:
            channel_meta["original_L_c"] = current_context_in.shape[1]
            channel_meta["C_c"] = current_context_in.shape[2]
            flat_channel_counts.append(0)  # Not stacked
        else:
            flat_channel_counts.append(0)

        if current_context_out is not None:
            channel_meta["original_L_co"] = current_context_out.shape[1]
            channel_meta["C_co"] = current_context_out.shape[2]
            flat_channel_counts.append(0)  # Not stacked
        else:
            flat_channel_counts.append(0)

        # Stack image_context_in_prev if provided
        if image_context_in_prev is not None:
            channel_meta["C_ic"] = image_context_in_prev.shape[2]
            # Reshape image_context_in: [B, 1, C_ic, D, H, W] -> [B, C_ic, D, H, W] for stacking
            reshaped_img_ctx_in = image_context_in_prev.reshape(B_main, -1, *spatial_shape_iter)
            inputs_to_stack.append(reshaped_img_ctx_in.to(self.device))
            flat_channel_counts.append(reshaped_img_ctx_in.shape[1])
        else:
            flat_channel_counts.append(0) # Placeholder if not present

        # Concatenate all inputs along the channel dimension
        stacked_inputs = torch.cat(inputs_to_stack, dim=1)

        # Memoize metadata for use in the nested predictor function
        memoized_flat_channel_counts = flat_channel_counts
        memoized_foreground_centers = foreground_patch_centers
        memoized_context_in = current_context_in
        memoized_context_out = current_context_out
        memoized_roi_size = roi_size
        memoized_level_data = level_data

        # Patch counter for inspection
        patch_counter = [0]  # Use list to make it mutable in closure

        # Define the predictor function that will be called by sliding_window_inference for each window
        def _predictor_for_sliding_window(data_window, network_l_param_inner):
            """
            This nested function is called by sliding_window_inference on each window (patch) of the stacked_inputs.
            It unpacks the window data back into individual model inputs and calls self.forward.

            MODIFIED: Samples context patches from foreground locations.
            """
            current_idx = 0
            # Extract target_window
            target_window = data_window[:, current_idx : current_idx + memoized_flat_channel_counts[0], ...]
            current_idx += memoized_flat_channel_counts[0]

            # Sample context patches from foreground (MODIFIED)
            actual_context_in = None
            actual_context_out = None
            sampled_context_coords = None
            if memoized_context_in is not None and memoized_foreground_centers is not None:
                # Sample patches from foreground locations
                ctx_in = memoized_context_in[0]  # [L, C, D, H, W]
                ctx_out = memoized_context_out[0]  # [L, C, D, H, W]

                # Track sampled coordinates for inspection
                if memoized_level_data is not None:
                    sampled_context_coords = []
                    for l_idx in range(len(memoized_foreground_centers)):
                        if len(memoized_foreground_centers[l_idx]) > 0:
                            center = random.choice(memoized_foreground_centers[l_idx])
                        else:
                            D, H, W = ctx_in.shape[2:]
                            D_roi, H_roi, W_roi = memoized_roi_size
                            center = (
                                random.randint(0, max(0, D - D_roi)),
                                random.randint(0, max(0, H - H_roi)),
                                random.randint(0, max(0, W - W_roi))
                            )
                        sampled_context_coords.append(center)

                    # Use tracked coordinates to sample
                    sampled_in = []
                    sampled_out = []
                    for l_idx, (d, h, w) in enumerate(sampled_context_coords):
                        D_roi, H_roi, W_roi = memoized_roi_size
                        patch_in = ctx_in[l_idx:l_idx+1, :, d:d+D_roi, h:h+H_roi, w:w+W_roi]
                        patch_out = ctx_out[l_idx:l_idx+1, :, d:d+D_roi, h:h+H_roi, w:w+W_roi]
                        sampled_in.append(patch_in)
                        sampled_out.append(patch_out)
                    sampled_ctx_in = torch.cat(sampled_in, dim=0)
                    sampled_ctx_out = torch.cat(sampled_out, dim=0)
                else:
                    # Normal sampling without tracking
                    sampled_ctx_in, sampled_ctx_out = self._sample_context_patches(
                        ctx_in,
                        ctx_out,
                        memoized_roi_size,
                        memoized_foreground_centers,
                        num_samples=1
                    )

                # Reshape for model: add batch dimension
                actual_context_in = sampled_ctx_in.unsqueeze(0)  # [1, L, C, D_roi, H_roi, W_roi]
                actual_context_out = sampled_ctx_out.unsqueeze(0)  # [1, L, C, D_roi, H_roi, W_roi]

            current_idx += memoized_flat_channel_counts[1]  # Skip context_in placeholder
            current_idx += memoized_flat_channel_counts[2]  # Skip context_out placeholder

            # Extract and reshape image_context_in_window if it was part of the stack
            actual_img_ctx_in = None
            actual_img_ctx_out = None
            if memoized_flat_channel_counts[3] > 0:
                img_ctx_in_window_flat = data_window[:, current_idx : current_idx + memoized_flat_channel_counts[3], ...]
                # Reshape to [B_window, 1, C_model_out, D_roi, H_roi, W_roi]
                # img_ctx_in_window_flat is [B_window, C_model_out, D_roi, H_roi, W_roi]
                actual_img_ctx_out = img_ctx_in_window_flat[:,0:1,:].unsqueeze(1)
                actual_img_ctx_in = img_ctx_in_window_flat[:,1:2,:].unsqueeze(1)


            if hasattr(self, 'concat_global_crop') and self.concat_global_crop:
                actual_context_in = torch.cat([actual_context_in, self.current_context_in_resized.to(self.device)], dim=1)
                actual_context_out = torch.cat([actual_context_out, self.current_context_out_resized.to(self.device)], dim=1)


            mask = self.forward(target_in=target_window,
                                context_in=actual_context_in,
                                context_out=actual_context_out,
                                image_context_in=actual_img_ctx_in,
                                image_context_out=actual_img_ctx_out,
                                l=network_l_param_inner
                               )

            # Store patch inspection data if enabled
            if memoized_level_data is not None:
                patch_id = patch_counter[0]
                patch_data = {
                    'patch_id': patch_id,
                    'target_in': target_window[0].detach().cpu(),  # [C, D_roi, H_roi, W_roi]
                    'prediction': mask[0].detach().cpu(),  # [C, D_roi, H_roi, W_roi]
                }

                if actual_context_in is not None:
                    patch_data['context_in'] = actual_context_in[0].detach().cpu()  # [L, C, D_roi, H_roi, W_roi]
                    patch_data['context_out'] = actual_context_out[0].detach().cpu()  # [L, C, D_roi, H_roi, W_roi]
                    patch_data['sampled_context_coords'] = sampled_context_coords  # List of (d, h, w) tuples

                memoized_level_data['patches'].append(patch_data)
                patch_counter[0] += 1

            # ----plot image
            if self.verbose:
                if actual_context_in is not None:
                    self.plot_fig(actual_context_in[0], actual_context_out[0], slice_ = 64, name = 'Context', colorbar = True)
                if actual_img_ctx_in is not None:
                    self.plot_pred_fig(target_window, actual_img_ctx_in[0], actual_img_ctx_out[0], slice_ = 64, name = 'PredImgInImgOut', colorbar = True)
                self.plot_pred_fig(target_window, mask, mask, slice_ = 64, name = 'PredOutOut', colorbar = True)

            # Call the model's forward pass with the unpacked and reshaped window inputs
            return mask

        # Perform sliding window inference using MONAI's utility
        output_pred = sliding_window_inference(
            inputs=stacked_inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=_predictor_for_sliding_window, # Custom predictor defined above
            overlap=overlap,
            mode=mode,
            sigma_scale=0.125, # Standard for Gaussian blending
            padding_mode=PytorchPadMode.CONSTANT,
            cval=0.0,
            sw_device=self.device, # Device for window operations
            device=self.device,    # Device for the final output
            progress=False,
            network_l_param_inner=forward_l_param # Pass the 'l' for forward to the predictor
        )

        # Append level data to inspection data if enabled
        if level_data is not None:
            self.inspection_data['levels'].append(level_data)

        return output_pred

    def autoregressive_inference(self, *args, **kwargs):
        """
        Override to use custom autoregressive inference with level tracking.
        """
        # Simply call our custom implementation that tracks levels
        return self._autoregressive_inference_with_level_tracking(*args, **kwargs)

    def _autoregressive_inference_with_level_tracking(self,
                                 target_in,
                                 context_in=None,
                                 context_out=None,
                                 level=None,
                                 forward_l_arg=3,
                                 sw_roi_size=(128,128,128),
                                 sw_overlap=0.1,
                                 sw_blend_mode=BlendMode.GAUSSIAN,
                                 sw_batch_size_val=1,
                                 intensity_clip = True,
                                 verbose = False):
        """
        Modified autoregressive inference that tracks levels for inspection.
        Based on parent's implementation but adds level tracking.
        """
        # Import required functions
        from torch.nn import functional as F
        import numpy as np

        if level is None:
            max_axis = torch.tensor(target_in.shape).max()
            level = int(np.ceil(np.log2(max_axis / 128)) + 1)

        self.verbose = verbose

        B_orig, C_t_orig, D_orig, H_orig, W_orig = target_in.shape

        # Input shape validation (same as parent)
        if context_in is not None and \
           (context_in.shape[0]!=B_orig or context_in.shape[3:]!=(D_orig,H_orig,W_orig)):
            raise ValueError("Target and context_in mismatch in batch or spatial dimensions.")
        if context_out is not None and \
           (context_out.shape[0]!=B_orig or context_out.shape[3:]!=(D_orig,H_orig,W_orig)):
            raise ValueError("Target and context_out mismatch in batch or spatial dimensions.")
        if not all(s > 0 for s in sw_roi_size):
             raise ValueError("sw_roi_size must have positive components.")

        original_spatial_shape_tuple = (D_orig, H_orig, W_orig)

        # Padding logic - manually implement to avoid parent class bug
        padding_applied = False
        padded_hires_spatial_shape_list = list(original_spatial_shape_tuple)
        if level > 1:
            divisor = 2**(level - 1)
            for i in range(3):
                if padded_hires_spatial_shape_list[i] % divisor != 0:
                    padded_hires_spatial_shape_list[i] = (padded_hires_spatial_shape_list[i] // divisor + 1) * divisor

        current_padded_target_in = target_in
        current_padded_context_in = context_in
        current_padded_context_out = context_out

        if tuple(padded_hires_spatial_shape_list) != original_spatial_shape_tuple:
            padding_applied = True
            pad_d = padded_hires_spatial_shape_list[0] - D_orig
            pad_h = padded_hires_spatial_shape_list[1] - H_orig
            pad_w = padded_hires_spatial_shape_list[2] - W_orig
            padding_amounts_torch = (0, pad_w, 0, pad_h, 0, pad_d)

            current_padded_target_in = F.pad(target_in, padding_amounts_torch, mode='constant', value=0)
            if context_in is not None:
                B_c, L_c, C_c, _, _, _ = context_in.shape
                reshaped_ctx_in = context_in.reshape(B_c*L_c, C_c, D_orig, H_orig, W_orig)
                padded_reshaped_ctx_in = F.pad(reshaped_ctx_in, padding_amounts_torch, mode='constant', value=0)
                current_padded_context_in = padded_reshaped_ctx_in.reshape(B_c, L_c, C_c, *padded_hires_spatial_shape_list)
            if context_out is not None:
                B_co, L_co, C_co, _, _, _ = context_out.shape
                reshaped_ctx_out = context_out.reshape(B_co*L_co, C_co, D_orig, H_orig, W_orig)
                padded_reshaped_ctx_out = F.pad(reshaped_ctx_out, padding_amounts_torch, mode='constant', value=0)
                current_padded_context_out = padded_reshaped_ctx_out.reshape(B_co, L_co, C_co, *padded_hires_spatial_shape_list)

        padded_hires_spatial_shape_tuple = tuple(padded_hires_spatial_shape_list)

        # Autoregressive loop
        image_context_feedback = None
        image_context_feedback_in = None
        final_prediction = None

        for l_loop in range(1, level + 1):
            # Determine spatial dimensions for current level (same as parent)
            if l_loop == level:
                current_iter_input_spatial_shape = list(padded_hires_spatial_shape_tuple)
            else:
                scale_val = 2**(level - l_loop)
                current_iter_input_spatial_shape = [s // scale_val for s in padded_hires_spatial_shape_list]

            # Prepare image context (same as parent)
            model_image_context_in = None
            model_image_context_in_in = None
            if image_context_feedback is not None:
                model_image_context_in = self._check_size_image_context(image_context_feedback, current_iter_input_spatial_shape)
            if image_context_feedback_in is not None:
                model_image_context_in_in = self._check_size_image_context(image_context_feedback_in, current_iter_input_spatial_shape)

            # Downsample inputs (same as parent)
            original_spatial_for_this_iter_step, input_target_to_slider, \
                input_context_in_to_slider, input_context_out_to_slider, \
                input_img_ctx_to_slider, perform_temporary_upsample = self._downsample_for_autoregressive(
                    current_iter_input_spatial_shape,
                    sw_roi_size,
                    current_padded_target_in,
                    current_padded_context_in,
                    current_padded_context_out,
                    model_image_context_in,
                    model_image_context_in_in,
                )

            # Perform inference - WITH LEVEL TRACKING
            current_level_pred_output = self._sliding_window_autoregressive_step(
                current_target_in=input_target_to_slider,
                current_context_in=input_context_in_to_slider,
                current_context_out=input_context_out_to_slider,
                image_context_in_prev=input_img_ctx_to_slider,
                roi_size=sw_roi_size,
                sw_batch_size=sw_batch_size_val,
                overlap=sw_overlap,
                mode=sw_blend_mode,
                forward_l_param=forward_l_arg,
                current_level=l_loop  # PASS LEVEL HERE
            )

            # Resample for next loop (same as parent)
            image_context_feedback, image_context_feedback_in, final_prediction = self._resample_image_for_next_loop(
                current_level_pred_output,
                input_target_to_slider,
                perform_temporary_upsample,
                original_spatial_for_this_iter_step,
                level,
                l_loop,
                padded_hires_spatial_shape_tuple,
                padded_hires_spatial_shape_list,
                intensity_clip
            )

        # Final cropping (same as parent)
        if padding_applied and final_prediction is not None:
            final_prediction = final_prediction[:, :, :D_orig, :H_orig, :W_orig]

        return final_prediction
