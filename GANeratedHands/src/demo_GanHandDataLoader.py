'''
watch nvidia-smi

source ~/.bashrc
source ????/bin/activate

python3 -u demo_GanHandDataLoader.py &> logs/demo_GanHandDataLoader.log &

'''
import os
import argparse

from common_utils import to_print, makeDir
from datasets.Base.utils import demo_data_loader

# ===========================================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_path', default="../datasets/GANeratedHands", help="Path to dataset")
ap.add_argument('--data_subset', default="noObject",
                choices=["noObject", "withObject", "all"])
args = ap.parse_args()

# ===========================================================================================
if __name__ == '__main__':

    to_print("PID = %d \n" % os.getpid())

    save_to = makeDir("results")
    demo_data_loader("GANeratedHands", args.dataset_path, args.data_subset, save_to, n=25)
