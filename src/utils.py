import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import os
from nilearn.image import resample_img

def display_img_and_label (img, labels, slices=None):
    """
    Display slices of an image and its labels
    """
    if slices is None:
        slices = [img.shape[2] // 4 * i for i in range(4)]
    plt.style.use('default')
    fig, axes = plt.subplots(len(slices), 2, figsize=(12,12))
    for i, ax in enumerate(axes):
        ax[0].imshow(img[:,:,slices[i]], cmap='gray')
        ax[1].imshow(labels[:,:,slices[i]], cmap='tab20')
        ax[0].set_title(f"Image slice {slices[i]}")
        ax[1].set_title(f"Labels slice {slices[i]}")
        ax[0].axis('off')
        ax[1].axis('off')
    plt.show()


def dice_coefficient(pred, target):
    """Calculate Dice coefficient between prediction and target masks."""
    smooth = 1e-5
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def clip_intensity(image, lower_percentile=2, upper_percentile=98):
    """
    Clip the intensity of a numpy array (image) to the specified percentiles.

    Parameters:
    - image: numpy array, the input image (could be a 2D, 3D, or any n-dimensional array).
    - lower_percentile: float, the lower percentile (default is 5).
    - upper_percentile: float, the upper percentile (default is 95).

    Returns:
    - clipped_image: numpy array with intensity clipped to the percentiles.
    """
    # Calculate the lower and upper percentile values
    lower_value = np.percentile(image, lower_percentile)
    upper_value = np.percentile(image, upper_percentile)

    # Clip the image intensities
    clipped_image = np.clip(image, lower_value, upper_value)

    return clipped_image

def load_seg_data(image_dir, label_dir, reference_shape=None, nb_max=None):
    images = []
    labels = []
    reference_affine = None
    
    # Get all image files and extract IDs
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_img.nii.gz')])
    if nb_max is not None and len(img_files) > nb_max:
        img_files = img_files[:nb_max]
        
    for idx, img_file in enumerate(img_files):
        # Extract ID from filename (e.g., '001_img.nii.gz' -> '001')
        file_id = img_file.split('_img.nii.gz')[0]
        
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, f'{file_id}_gt_all.nii.gz')

        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)

        # Set reference on first image
        if idx == 0:
            if reference_shape is None:
                # Use first image's shape and affine as reference
                reference_shape = img_nii.shape
                reference_affine = img_nii.affine
            else:
                # Create a simple affine for the target shape
                # Use isotropic 1mm voxels centered at origin
                reference_affine = np.eye(4)
                reference_affine[:3, 3] = -np.array(reference_shape) / 2
        
        # Resample all images (including first) to reference
        img_resampled = resample_img(
            img_nii, 
            target_affine=reference_affine, 
            target_shape=reference_shape,
            interpolation='continuous'
        )
        img_data = img_resampled.get_fdata()
        img_data = clip_intensity(img_data)
        
        label_resampled = resample_img(
            label_nii,
            target_affine=reference_affine,
            target_shape=reference_shape,
            interpolation='nearest'
        )
        label_data = label_resampled.get_fdata()

        images.append(img_data[np.newaxis, ...]) 
        labels.append(label_data[np.newaxis, ...])

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)
    
    return images, labels