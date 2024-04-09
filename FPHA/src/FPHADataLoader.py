import os
import glob
import matplotlib.pyplot as plt
import numpy as np

from datasets.Base.DataLoader import DataLoader

# ==========================================================================================================
class FPHADataLoader(DataLoader):

    def __init__(self, data_path, subset="", batch_size=1, shuffle=True):

        self.data_path = os.path.join(data_path, "Video_files")

        super(FPHADataLoader, self).__init__(name="FPHA", data_path=data_path, subset=subset,
                                                batch_size=batch_size, shuffle=shuffle)        

    # def getFileBatch(self, ind):
    #     return self.files[self._batches[ind]]

    def getBatch(self, ind, load_img=True):
        frames = []
        coords = []
        paths = []
        names = []
        for ij in self._batches[ind]:
            # --- load image ----
            if load_img:
                image = plt.imread(self.files[ij]).astype(np.uint8)
                if np.max(image) <= 1:
                     image = (image * 255).astype(np.uint8)
                frames.append(image)
            # --- load joints coordinates ----
            path = self.files[ij].split("/")
            skeleton_path = os.path.join(self.main_path, 'Hand_pose_annotation', path[-5], path[-4], path[-3])
            coords.append(self._getCoordintanes(skeleton_path, int(self.files[ij].split("/")[-1][-9:-5])))
            # --------------------------
            paths.append(os.path.join(path[-5], path[-4], path[-3]))
            names.append(self.files[ij].split("/")[-1])

        return frames, coords, paths, names

    def _getFileList(self):
        List = []
        if self.subset == "all":
            subjects = glob.glob(os.path.join(self.data_path, "*/"))
            subjects.sort()
            for subject in subjects:
                subject = subject.split("/")[-2]
                List.extend(self._getSubjectFileList(subject))
        else:
            List.extend(self._getSubjectFileList(self.subset))

        return List

    def _getSubjectFileList(self, subset):
        List = []

        subsets = list(glob.glob(os.path.join(self.data_path, subset, "*/")))
        subsets.sort()
        for ss in subsets:
            subsubsets = list(glob.glob(os.path.join(ss, "*/")))
            subsubsets.sort()
            for sss in subsubsets:
                path = os.path.join(sss, "color")
                frames = glob.glob(os.path.join(sss, "color", "*.jpeg"))
                frames.sort()
                List.extend(frames)
        return List

    def _get_skeleton(self, skeleton_path, frame_idx):
        skeleton_vals = np.loadtxt(os.path.join(skeleton_path, "skeleton.txt"))
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)[frame_idx]

        return skeleton

    def _getCoordintanes(self, skeleton_path, frame_idx):
        reorder_idx = np.array([
            0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
            20
        ])
        cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594,
              25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807,
              3.902], [0, 0, 0, 1]])
        cam_intr = np.array([[1395.749023, 0, 935.732544],
                             [0, 1395.749268, 540.681030], [0, 0, 1]])

        skel = self._get_skeleton(skeleton_path, frame_idx)[reorder_idx]

        # Apply camera extrinsic to hand skeleton
        skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        skel_camcoords = cam_extr.dot(
            skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
        skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
        skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

        return skel_proj
