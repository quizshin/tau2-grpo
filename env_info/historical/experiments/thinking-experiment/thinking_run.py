from pathlib import Path
import sys,runpy
import tau3_grpo.training.sft.dataset as dataset
from thinking_dataset import build
dataset.build_supervised_example=build
runpy.run_path(str(Path(__file__).with_name('train_thinking.py')),run_name='__main__')
