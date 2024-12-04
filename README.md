
Request cluster ressources 
srun -p ml_gpu-rtx2080 --time=3:00:00 --pty bash

activate environment 
source .venv/bin/activate

Evaluating the SegGPT model on an in context segmentation task.
https://github.com/baaivision/Painter/blob/main/SegGPT/SegGPT_inference/README.md

run original script via command line:
python seggpt_inference.py --input_image examples/brain_imgs/slice_31_img.jpg --prompt_image examples/brain_imgs/slice_30_img.jpg --prompt_target examples/brain_imgs/slice_30_labels.png --output_dir ./ --device cpu

export the path to the SegGPT_inference folder in order to import the necessary modules:
export PYTHONPATH="/home/ndirt/dev/radiology/Painter/SegGPT/SegGPT_inference:$PYTHONPATH"
"""