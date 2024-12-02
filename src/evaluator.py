import wandb
import numpy as np
from pathlib import Path
import time
from PIL import Image
from abc import ABC, abstractmethod

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

    
# Later, when SegGPT is ready, create a SegGPT wrapper:
class SegGPTModel(SegmentationModel):
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def predict(self, input_img, prompt_img, prompt_mask):
        # Implement SegGPT-specific processing here
        # This method should handle all the preprocessing and postprocessing
        pass

class SegmentationEvaluator:
    def __init__(self, model: SegmentationModel, project_name="segmentation-evaluation", 
                 entity=None, config=None):
        """
        Initialize evaluator with any segmentation model.
        
        Args:
            model: Instance of SegmentationModel
            project_name: W&B project name
            entity: W&B entity/username
            config: Dictionary of configuration parameters to log
        """
        self.model = model
        
        # Initialize W&B run
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            config=config or {
                "model_type": model.__class__.__name__,
            }
        )
        
        self.results_table = wandb.Table(columns=[
            "test_image", "prompt_image", "dice_score", 
            "inference_time", "prediction"
        ])

    def dice_coefficient(self, pred, target):
        """Calculate Dice coefficient.
        
        Args:
            pred: Prediction mask (PIL Image or numpy array)
            target: Target mask (PIL Image or numpy array)
        """
        # Convert to PIL if numpy arrays
        if isinstance(pred, np.ndarray):
            pred = Image.fromarray(pred.astype(np.uint8))
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target.astype(np.uint8))
        
        # Ensure same size
        target_size = target.size
        pred = pred.resize(target_size, Image.NEAREST)
        
        # Convert to binary numpy arrays
        pred = np.array(pred).astype(bool)
        target = np.array(target).astype(bool)
        
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Calculate Dice
        smooth = 1e-5
        intersection = (pred & target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def evaluate_pair(self, input_path, prompt_path, prompt_mask_path):
        """Evaluate a single test case."""
        # Load images
        input_img = Image.open(input_path).convert('RGB')
        prompt_img = Image.open(prompt_path).convert('RGB')
        prompt_mask = Image.open(prompt_mask_path).convert('L')  # Load as grayscale
        
        # Run inference and time it
        start_time = time.time()
        pred_mask = self.model.predict(input_img, prompt_img, prompt_mask)
        inference_time = time.time() - start_time
        
        return pred_mask, inference_time

    def evaluate_dataset(self, dataset_path, results_dir=None):
        """Evaluate model on dataset."""
        dataset_path = Path(dataset_path)
        results_dir = Path(results_dir) if results_dir else None
        
        image_files = sorted(list((dataset_path / 'images').glob('*.jpg')))
        results = []
        
        # Track progress with wandb
        with wandb.init() as run:
            for i, test_image in enumerate(image_files):
                image_metrics = {
                    'test_image': test_image.name,
                    'dice_scores': [],
                    'inference_times': []
                }
                
                # Load ground truth mask once
                test_mask_path = dataset_path / 'masks' / test_image.with_suffix('.png').name
                true_mask = Image.open(test_mask_path).convert('L')  # Load as grayscale
                
                # Use each other image as prompt
                for j, prompt_image in enumerate(image_files):
                    if i == j:  # Skip using image as its own prompt
                        continue
                    
                    prompt_mask_path = dataset_path / 'masks' / prompt_image.with_suffix('.png').name
                    
                    try:
                        # Run prediction
                        pred_mask, inf_time = self.evaluate_pair(
                            str(test_image),
                            str(prompt_image),
                            str(prompt_mask_path)
                        )
                        
                        # Calculate metrics
                        dice = self.dice_coefficient(pred_mask, true_mask)
                        
                        # Store results
                        image_metrics['dice_scores'].append(dice)
                        image_metrics['inference_times'].append(inf_time)
                        
                        # Log to wandb
                        wandb.log({
                            'batch/dice_score': dice,
                            'batch/inference_time': inf_time,
                            'batch/test_image': test_image.name,
                            'batch/prompt_image': prompt_image.name
                        })
                        
                        # Save prediction if requested
                        if results_dir:
                            pred_path = results_dir / f"{test_image.stem}_prompt{j}.png"
                            if isinstance(pred_mask, np.ndarray):
                                pred_mask = Image.fromarray(pred_mask.astype(np.uint8))
                            pred_mask.save(pred_path)
                    
                    except Exception as e:
                        print(f"Error processing {test_image.name} with prompt {prompt_image.name}: {str(e)}")
                        continue
                
                # Only compute summary if we have results
                if image_metrics['dice_scores']:
                    image_summary = {
                        'test_image': test_image.name,
                        'mean_dice': np.mean(image_metrics['dice_scores']),
                        'std_dice': np.std(image_metrics['dice_scores']),
                        'mean_inference_time': np.mean(image_metrics['inference_times']),
                        'std_inference_time': np.std(image_metrics['inference_times'])
                    }
                    results.append(image_summary)
                    
                    # Log summary to wandb
                    wandb.log({
                        'image_summary/mean_dice': image_summary['mean_dice'],
                        'image_summary/mean_inference_time': image_summary['mean_inference_time']
                    })
            
            if results:
                # Calculate overall metrics
                overall_metrics = {
                    'overall/mean_dice': np.mean([r['mean_dice'] for r in results]),
                    'overall/std_dice': np.std([r['mean_dice'] for r in results]),
                    'overall/mean_inference_time': np.mean([r['mean_inference_time'] for r in results]),
                    'overall/std_inference_time': np.std([r['mean_inference_time'] for r in results])
                }
                
                # Log final metrics
                wandb.log(overall_metrics)
                
                return results, overall_metrics
            else:
                print("No valid results were generated!")
                return None, None