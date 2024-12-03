import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import torch 
from seggpt_engine import run_one_image
import torch.nn.functional as F

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
    """Wrapper for SegGPT model following the SegmentationModel interface."""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize SegGPT model wrapper.
        
        Args:
            model: Loaded SegGPT model instance
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.image_size = (448, 448)  # SegGPT's expected input size
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        
    def preprocess_images(self, input_img, prompt_img, prompt_mask):
        """Preprocess images following SegGPT's requirements."""
        # Convert to PIL if needed
        if isinstance(input_img, np.ndarray):
            input_img = Image.fromarray(input_img)
        if isinstance(prompt_img, np.ndarray):
            prompt_img = Image.fromarray(prompt_img)
        if isinstance(prompt_mask, np.ndarray):
            prompt_mask = Image.fromarray(prompt_mask.astype(np.uint8))
            
        # Store original size for later
        original_size = input_img.size
            
        # Resize to SegGPT's expected size
        res, hres = self.image_size
        input_img_resized = np.array(input_img.resize((res, hres))) / 255.
        prompt_img_resized = np.array(prompt_img.resize((res, hres))) / 255.
        prompt_mask_resized = np.array(prompt_mask.resize((res, hres), Image.NEAREST)) / 255.
        
        # Concatenate images as expected by SegGPT
        img = np.concatenate((prompt_img_resized, input_img_resized), axis=0)
        tgt = np.concatenate((prompt_mask_resized, prompt_mask_resized), axis=0)
        
        # Normalize
        img = (img - self.imagenet_mean) / self.imagenet_std
        tgt = (tgt - self.imagenet_mean) / self.imagenet_std
        
        return img, tgt, original_size

    def predict(self, input_img, prompt_img, prompt_mask):
        """
        Predict segmentation mask using SegGPT.
        
        Args:
            input_img: PIL Image or numpy array of image to segment
            prompt_img: PIL Image or numpy array of prompt image
            prompt_mask: PIL Image or numpy array of prompt mask
            
        Returns:
            PIL Image: Predicted segmentation mask
        """
        # Convert inputs to PIL images if they're numpy arrays
        if isinstance(input_img, np.ndarray):
            input_img = Image.fromarray(input_img)
        
        # Store original input for blending
        input_array = np.array(input_img)
        
        # Preprocess images
        img, tgt, original_size = self.preprocess_images(input_img, prompt_img, prompt_mask)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        tgt = np.expand_dims(tgt, axis=0)
        
        # Set random seed for reproducibility
        torch.manual_seed(2)
        
        # Run inference
        output = run_one_image(img, tgt, self.model, self.device)
        
        # Resize to original dimensions
        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[original_size[1], original_size[0]],
            mode='nearest'
        ).permute(0, 2, 3, 1)[0].cpu().numpy()
        
        # Blend with original image following SegGPT's approach
        blended_output = (input_array * (0.6 * output / 255 + 0.4)).astype(np.uint8)
        
        return Image.fromarray(blended_output)

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