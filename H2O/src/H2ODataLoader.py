import os
import numpy as np
import matplotlib.pyplot as plt

from datasets.Base.DataLoader import DataLoader

# ==========================================================================================================
class H2ODataLoader(DataLoader):

    def __init__(self, data_path, envs=None, subset="", batch_size=1, shuffle=True):

        self.data_path = os.path.join(data_path, "subjects")
        self.envs = ["h1", "h2", "k1", "k2", "o1", "o2"] if envs is None else envs

        super(H2ODataLoader, self).__init__(name="H2O", data_path=data_path, subset=subset,
                                                batch_size=batch_size, shuffle=shuffle)

    def getBatch(self, ind, load_img=True):
        frames = []
        coords = []
        paths  = []
        names  = []
        for ij in self._batches[ind]:
            # --- load image ----
            if self.files[ij] == "":
                continue
            file = (self.files[ij] % "rgb") + ".png"

            if load_img:
                image = plt.imread( os.path.join(self.data_path, file))
                if np.max(image) <= 1:
                     image = (image * 255).astype(np.uint8)
                frames.append(image)
            # --- load joints coordinates ----
            file = (self.files[ij] % "hand_pose") + ".txt"
            cam_intr = self._loadCameraIntrinsics(os.path.join(self.data_path, self.files[ij].split("%s")[0]))
            coords.append(self._getCoordintanes(os.path.join(self.data_path, file), cam_intr))
            # --------------------------
            paths.append(os.path.join(self.files[ij].split("%s")[0])[:-6])
            names.append(self.files[ij].split("%s")[1][1:])

        return frames, coords, paths, names

    def getByName(self, key, name):
        frames = []
        coords = []
        paths  = []
        names  = []

        pref = os.path.join(key.replace("_", "/"), 'cam4')
        file = os.path.join(self.data_path, pref, 'rgb', '{}.png'.format(name))
        image = plt.imread( os.path.join(self.data_path, file))
        if np.max(image) <= 1:
             image = (image * 255).astype(np.uint8)
        frames.append(image)

        # --- load joints coordinates ----
        file = os.path.join(self.data_path, pref, 'hand_pose', '{}.txt'.format(name))
        cam_intr = self._loadCameraIntrinsics(os.path.join(self.data_path, pref))
        coords.append(self._getCoordintanes(os.path.join(self.data_path, file), cam_intr))
        # --------------------------
        paths.append(pref[:-5])
        names.append(name)

        return frames, coords, paths, names


    def _getFileList(self):
        List = []
        if self.subset == "all":
            subsets = ["train", "validation", "test"]
            for subset in subsets:
                List.extend(np.loadtxt(os.path.join(self.main_path, "label_split", "pose_%s.txt" % subset)))
        else:
            List.extend(self._loadFilesList(os.path.join(self.main_path, "label_split", "pose_%s.txt" % self.subset)))

        return List

    def _loadFilesList(self, file_path, n=1):
        contents = []

        with open(file_path) as f:
            str = f.read().replace("rgb", "%s").replace(".png", "")
            contents = str.split("\n")

        return contents

    def _loadCameraIntrinsics(self, path):
        fx, fy, cx, cy, width, height = np.loadtxt(os.path.join(path, "cam_intrinsics.txt"))
        cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return cam_intr

    def _getCoordintanes(self, file_path, cam_intr, n=3):
        coords = []
        with open(file_path) as f:
            contents = f.read()
            l = [float(idx) for idx in contents.split(" ")[:-1]]
            del l[0]
            del l[63]
            coords = np.array([l[i:i + n] for i in range(0, len(l), n)])

        skel_hom2d = np.array(cam_intr).dot(coords.transpose()).transpose()
        skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

        return np.array(skel_proj).reshape(2,-1, 2)
