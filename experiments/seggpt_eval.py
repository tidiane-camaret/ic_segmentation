""""
Request cluster ressources 
srun -p ml_gpu-rtx2080 --time=3:00:00 --pty bash

Evaluating the SegGPT model on an in context segmentation task.
https://github.com/baaivision/Painter/blob/main/SegGPT/SegGPT_inference/README.md

run original script via command line:
python seggpt_inference.py --input_image examples/brain_imgs/slice_31_img.jpg --prompt_image examples/brain_imgs/slice_30_img.jpg --prompt_target examples/brain_imgs/slice_30_labels.png --output_dir ./ --device cpu

export the path to the SegGPT_inference folder in order to import the necessary modules:
export PYTHONPATH="/home/ndirt/dev/radiology/Painter/SegGPT/SegGPT_inference:$PYTHONPATH"
"""

import os
import argparse

import torch
import numpy as np

import models_seggpt
from PIL import Image
import torch.nn.functional as F

from seggpt_engine import run_one_image

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='/home/ndirt/dev/radiology/Painter/SegGPT/SegGPT_inference/seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default='data/slice_30_img.jpg')
    parser.add_argument('--input_video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num_frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default='data/slice_31_img.jpg')
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default='data/slice_31_labels.png')
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='data/')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

def inference_image(model, device, img_path, img2_paths, tgt2_paths, out_path):
    res, hres = 448, 448

    image = Image.open(img_path).convert("RGB")
    input_image = np.array(image)
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.

    image_batch, target_batch = [], []
    for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)
    
        assert img.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)
    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), 
        size=[size[1], size[0]], 
        mode='nearest',
    ).permute(0, 2, 3, 1)[0].numpy()
    output = Image.fromarray((input_image * (0.6 * output / 255 + 0.4)).astype(np.uint8))
    
    return output

if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')
    args.prompt_image = [args.prompt_image]
    args.prompt_target = [args.prompt_target]

    img_name = os.path.basename(args.input_image)
    out_path = os.path.join(args.output_dir, "output_" + '.'.join(img_name.split('.')[:-1]) + '.png')

    output = inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path)
    output.save(out_path)
    print('Finished.')
