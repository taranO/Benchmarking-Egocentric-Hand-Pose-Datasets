import os
import cv2
import json
import numpy as np
from datetime import datetime

from datasets.Base.dataconfig import POSE_PAIRS, getDataLoader
# =============================================================================

def demo_data_loader(dataset, dataset_path, data_subset, save_to, n=0):

    dataloader = getDataLoader(dataset, dataset_path, data_subset)
    # ..............................................................
    colors = [(255, 0, 0), (255, 0, 0)]
    n = n if n > 0 else dataloader.n_batches
    # ..............................................................
    for i in range(n):
        # Local batches and labels
        frames, coords, paths, names = dataloader.getBatch(i)

        for j in range(len(frames)):
            frame = frames[j]
            joints = coords[j]
            if isinstance(joints, list):
                joints = np.asarray(joints)
            joints = joints.astype(int)
            path = paths[j].replace("/", "_")

            to_print(path + "_" + names[j])
            if joints.shape[0] == 21:
                joints = [joints]

            for hand in range(len(joints)):
                if len(joints[hand]):
                    for pair in POSE_PAIRS:
                        partA = pair[0]
                        partB = pair[1]

                        cv2.line(frame, (joints[hand][partA][0], joints[hand][partA][1]),
                                 (joints[hand][partB][0], joints[hand][partB][1]), colors[hand], 2)

            cv2.imwrite(os.path.join(save_to, path + "_" + names[j] + ".png"), frame[:, :, ::-1])

def to_print(message):
    now = datetime.now()
    print(f"{now.strftime('%d.%m.%Y %H:%M:%S')}: {message}")

def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def saveAsJSON(data, save_to):
    json_data = json.dumps(data)

    with open(save_to, 'w') as f:
        f.write(json_data)

def loadJSON(load_from):

    with open(load_from) as f:
        data = f.read()

    return json.loads(data)
