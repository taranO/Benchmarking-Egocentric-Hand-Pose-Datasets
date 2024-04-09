'''
watch nvidia-smi

source ~/.bashrc
source ~/Python/envs/database/bin/activate
cd ~/Python/datasets/H2O/src

python3 -u demo_h2oDataLoader.py &> demo_h2oDataLoader.log &

'''
import os
import argparse
import matplotlib.pyplot as plt

from common_utils import to_print, makeDir
from datasets.Base.utils import demo_data_loader

# ===========================================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_path', default="../datasets/H2O/", help="Path to dataset")
ap.add_argument('--data_subset', default="test",
                choices=["all", "train", "val", "test"])
args = ap.parse_args()

# ===========================================================================================
if __name__ == '__main__':

    to_print("PID = %d \n" % os.getpid())

    save_to = makeDir("results")
    demo_data_loader("H2O", args.dataset_path, args.data_subset, save_to, n=25)
