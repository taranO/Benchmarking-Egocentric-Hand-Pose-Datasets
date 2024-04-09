import os
import argparse

from common_utils import to_print, makeDir
from datasets.Base.utils import demo_data_loader

# ===========================================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_path', default="../datasets/Ego3DHands", help="Path to dataset")
ap.add_argument('--data_subset', default="static",
                choices=["static", "dynamic", "all"])
args = ap.parse_args()

# ===========================================================================================
if __name__ == '__main__':

    to_print("PID = %d \n" % os.getpid())
    save_to = makeDir("results")
    demo_data_loader("Ego3DHands", args.dataset_path, args.data_subset, save_to, n=25)
