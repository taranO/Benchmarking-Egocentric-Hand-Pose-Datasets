'''
watch nvidia-smi

source ~/.bashrc
source ????/bin/activate
cd datasets/SynthHands/src/

python3 -u demo_synthHandsDataLoader.py &> logs/demo_synthHandsDataLoader.log &

'''
import argparse
from SynthHandsDataLoader import *

from common_utils import to_print, makeDir
from datasets.Base.utils import demo_data_loader

# ===========================================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_path', default="../datasets/SynthHands", help="Path to dataset")
ap.add_argument('--data_subset', default="withObject",
                choices=["withObject", "noObject", "all"])
args = ap.parse_args()

# ===========================================================================================
if __name__ == '__main__':

    to_print("PID = %d \n" % os.getpid())

    save_to = makeDir("results")
    demo_data_loader("SynthHands", args.dataset_path, args.data_subset, save_to, n=25)

