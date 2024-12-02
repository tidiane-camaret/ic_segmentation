import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src import SegmentationEvaluator, CopyPromptModel

# Use the baseline CopyPromptModel
baseline_model = CopyPromptModel()
evaluator = SegmentationEvaluator(
    model=baseline_model,
    project_name="ic_segmentation",
    #entity="your_username"
)

# Run evaluation
results, metrics = evaluator.evaluate_dataset(
    dataset_path="data/mni_t1",
    #results_dir="results"
)

