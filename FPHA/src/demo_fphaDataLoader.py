'''
watch nvidia-smi

source ~/.bashrc
source ????/bin/activate

python3 -u demo_fphaDataLoader.py &> logs/demo_fphaDataLoader.log &

'''
import os
import argparse

from common_utils import to_print, makeDir
from datasets.Base.utils import demo_data_loader

# ===========================================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_path', default="../datasets/FPHA", help="Path to dataset")
ap.add_argument('--data_subset', default="Subject_1", choices=["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"],
                help="Dataset subset")
args = ap.parse_args()
# ===========================================================================================
if __name__ == '__main__':

    to_print("PID = %d \n" % os.getpid())
    save_to = makeDir("results")
    demo_data_loader("FPHA", args.dataset_path, args.data_subset, save_to, n=25)

