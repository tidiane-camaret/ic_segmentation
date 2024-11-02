import matplotlib.pylab as plt
import numpy as np
import nibabel as nib


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