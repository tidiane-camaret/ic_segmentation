import wandb
import numpy as np
from pathlib import Path
import time
from PIL import Image

from src.segmentation_models import SegmentationModel


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
        """Calculate Dice coefficient ensuring same sizes."""
        # Convert to numpy arrays
        pred = np.array(pred)
        target = np.array(target)
        
        # If prediction is RGB, convert to grayscale by taking first channel
        if len(pred.shape) == 3:
            pred = pred[:, :, 0]  # Take first channel
        
        # Convert to boolean
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Calculate Dice
        smooth = 1e-5
        intersection = (pred & target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def create_comparison_image(self, original_img, true_mask, pred_mask):
        """Create a side-by-side comparison image.
        
        Args:
            original_img: Path or PIL Image of the original image
            true_mask: Path or PIL Image of the ground truth mask
            pred_mask: PIL Image of the predicted mask
        """
        # Load images if paths are provided
        if isinstance(original_img, (str, Path)):
            original_img = Image.open(original_img).convert('RGB')
        if isinstance(true_mask, (str, Path)):
            true_mask = Image.open(true_mask).convert('L')
        
        # Convert masks to RGB for visualization
        true_mask_rgb = Image.merge('RGB', (true_mask, true_mask, true_mask))
        if pred_mask.mode != 'RGB':
            pred_mask_rgb = Image.merge('RGB', (pred_mask, pred_mask, pred_mask))
        else:
            pred_mask_rgb = pred_mask
        
        # Ensure all images are the same size
        width, height = original_img.size
        true_mask_rgb = true_mask_rgb.resize((width, height), Image.NEAREST)
        pred_mask_rgb = pred_mask_rgb.resize((width, height), Image.NEAREST)
        
        # Create a new image with three images side by side
        new_width = width * 3
        comparison = Image.new('RGB', (new_width, height))
        
        # Paste the images
        comparison.paste(original_img, (0, 0))
        comparison.paste(true_mask_rgb, (width, 0))
        comparison.paste(pred_mask_rgb, (width * 2, 0))
        
        return comparison
        
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
                    
                    #try:
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
                    
                    comparison = self.create_comparison_image(
                        test_image,
                        test_mask_path,
                        pred_mask
                    )
                    
                    # Log to wandb
                    wandb.log({
                        'batch/dice_score': dice,
                        'batch/inference_time': inf_time,
                        'batch/test_image': test_image.name,
                        'batch/prompt_image': prompt_image.name,
                        'visualizations': wandb.Image(
                            comparison,
                            caption=f"Left: Original, Middle: Ground Truth, Right: Prediction (Dice: {dice:.3f})"
                        )
                    })
                    
                    # Save prediction if requested
                    if results_dir:
                        pred_path = results_dir / f"{test_image.stem}_prompt{j}.png"
                        if isinstance(pred_mask, np.ndarray):
                            pred_mask = Image.fromarray(pred_mask.astype(np.uint8))
                        pred_mask.save(pred_path)
                    """
                    except Exception as e:
                        print(f"Error processing {test_image.name} with prompt {prompt_image.name}: {str(e)}")
                        continue
                    """
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