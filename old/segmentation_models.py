import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from copy import deepcopy

import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from transformers import AutoImageProcessor

from seggpt_engine import run_one_image

class SegmentationModel(ABC):
    """Abstract base class for segmentation models."""
    
    @abstractmethod
    def predict(self, input_img, prompt_img, prompt_mask):
        """Predict segmentation mask for input image given a prompt.
        
        Args:
            input_img: PIL Image or numpy array of image to segment
            prompt_img: PIL Image or numpy array of prompt image
            prompt_mask: PIL Image or numpy array of prompt mask
            
        Returns:
            numpy array: Predicted segmentation mask
        """
        pass

class CopyPromptModel(SegmentationModel):
    """Baseline model that simply returns the prompt mask."""
    
    def predict(self, input_img, prompt_img, prompt_mask):
        """Return the prompt mask resized to input image size."""
        # Ensure input_img is PIL Image
        if isinstance(input_img, np.ndarray):
            input_img = Image.fromarray(input_img)
            
        # Ensure prompt_mask is PIL Image
        if isinstance(prompt_mask, np.ndarray):
            prompt_mask = Image.fromarray(prompt_mask.astype(np.uint8))
            
        # Get target size
        target_size = input_img.size  # PIL size is (width, height)
        
        # Resize prompt mask to match input image size
        return prompt_mask.resize(target_size, Image.NEAREST)

class SegGPTModel(SegmentationModel):
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def predict(self, input_img, prompt_img, prompt_mask):
        # Fixed parameters
        res, hres = 448, 448
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        # Store original size for later
        original_size = input_img.size

        # Process images
        image = input_img.convert("RGB")
        img2 = prompt_img.convert("RGB")
        tgt2 = prompt_mask.convert("RGB")

        # Resize and convert to numpy
        image = np.array(image.resize((res, hres))) / 255.
        img2 = np.array(img2.resize((res, hres))) / 255.
        tgt2 = np.array(tgt2.resize((res, hres), Image.NEAREST)) / 255.

        # Create concatenated inputs
        tgt = np.concatenate((tgt2, tgt2), axis=0)
        img = np.concatenate((img2, image), axis=0)
        
        # Normalize
        img = (img - imagenet_mean) / imagenet_std
        tgt = (tgt - imagenet_mean) / imagenet_std

        # Add batch dimension
        img = np.stack([img], axis=0)
        tgt = np.stack([tgt], axis=0)

        # Run inference
        torch.manual_seed(2)
        output = run_one_image(img, tgt, self.model, self.device)

        # Resize output to original image size
        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2), 
            size=original_size[::-1],  # PIL size is (width, height), we need (height, width)
            mode='nearest',
        ).permute(0, 2, 3, 1)[0].cpu().numpy()
        
        # Convert to binary mask
        output = output.mean(axis=-1)  # Average across RGB channels if present
        output = (output > 0.5).astype(np.uint8) * 255  # Threshold and convert to 0-255 range
        
        # Return binary mask
        return Image.fromarray(output, mode='L')  # 'L' mode for single-channel grayscale

def load_seggpt_model(checkpoint_path, model_arch='seggpt_vit_large_patch16_input896x448', 
                     seg_type='instance', device='cuda'):
    """
    Helper function to load a SegGPT model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_arch: Model architecture name
        seg_type: Type of segmentation ('instance' or 'semantic')
        device: Device to load model on
        
    Returns:
        SegGPTModel instance
    """
    # Import here to avoid dependency if not using SegGPT
    import models_seggpt
    
    # Build model
    model = getattr(models_seggpt, model_arch)()
    model.seg_type = seg_type
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model = model.to(device)
    
    return SegGPTModel(model, device)

class SegmentationDataset(Dataset):
    """Dataset for segmentation image-mask pairs."""
    def __init__(self, images, masks, image_processor):
        """
        Args:
            images: List of PIL Images
            masks: List of PIL Images (masks)
            image_processor: HuggingFace image processor
        """
        assert len(images) == len(masks), "Number of images and masks must match"
        self.image_processor = image_processor
        self.processed_pairs = []
        
        # Process all image-mask pairs at initialization
        for img, mask in zip(images, masks):
            inputs = self.image_processor(img, mask, return_tensors="pt")
            self.processed_pairs.append({
                "pixel_values": inputs.pixel_values.squeeze(0),
                "labels": inputs.labels.squeeze(0)
            })

    def __len__(self):
        return len(self.processed_pairs)

    def __getitem__(self, idx):
        return self.processed_pairs[idx]

class FinetunedModel(SegmentationModel):
    def __init__(self, model, device='cuda', num_epochs=50):
        self.base_model = model
        self.device = device
        self.num_epochs = num_epochs
        
        self.image_processor = AutoImageProcessor.from_pretrained(
            "nvidia/mit-b0",
            reduce_labels=True
        )

    def finetune(self, images, masks):
        """
        Finetune model on image-mask pairs.
        
        Args:
            images: List of PIL Images or single PIL Image
            masks: List of PIL Images or single PIL Image
        """
        # Convert single images to lists
        if not isinstance(images, list):
            images = [images]
        if not isinstance(masks, list):
            masks = [masks]
            
        # Create dataset
        train_dataset = SegmentationDataset(images, masks, self.image_processor)

        training_args = TrainingArguments(
            output_dir="temp_finetuning",
            learning_rate=6e-5,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=1,
            save_strategy="no",
            remove_unused_columns=False,
            logging_steps=10,
            report_to="none"  # Disable wandb logging for finetuning
        )

        # Clone base model for this specific finetuning
        model = deepcopy(self.base_model)
        model.to(self.device)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        return model

    def predict(self, input_img, prompt_img, prompt_mask):
        """Finetune on prompt and predict on input image."""
        finetuned_model = self.finetune(prompt_img, prompt_mask)
        
        # Prepare input for inference
        encoding = self.image_processor(input_img, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = finetuned_model(pixel_values=pixel_values)
            logits = outputs.logits.cpu()

        # Post-process
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=input_img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Convert to binary mask
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        pred_seg = (pred_seg > 0).numpy().astype(np.uint8) * 255
        
        return Image.fromarray(pred_seg, mode='L')