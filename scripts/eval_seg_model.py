import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluator import SegmentationEvaluator
from src.segmentation_models import CopyPromptModel, load_seggpt_model


# Use the baseline CopyPromptModel
baseline_model = CopyPromptModel()

# Load the SegGPT model
seggpt_model = load_seggpt_model(
    checkpoint_path='/home/ndirt/dev/radiology/Painter/SegGPT/SegGPT_inference/seggpt_vit_large.pth',
    device='cuda'
)
    
evaluator = SegmentationEvaluator(
    model=seggpt_model, #, #baseline_model, 
    project_name="ic_segmentation",
)

# Run evaluation
results, metrics = evaluator.evaluate_dataset(
    dataset_path="data/mni_t1",
    #results_dir="results"
)

