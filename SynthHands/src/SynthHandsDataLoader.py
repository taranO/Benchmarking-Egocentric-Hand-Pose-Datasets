import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

from datasets.Base.DataLoader import DataLoader

# ==========================================================================================================
class SynthHandsDataLoader(DataLoader):

    def __init__(self, data_path, subset="", batch_size=1, shuffle=True, type='depth'):

        self.data_path = os.path.join(data_path, "data")

        if type == 'depth':
            self.camera_intrinsics = np.array([[475.62, 0, 311.125],
                                               [0, 475.62, 245.965],
                                               [0, 0, 1]])
            self.__file_type = '_color_on_depth.png'
        elif type == 'color':
            self.camera_intrinsics = np.array([[617.173, 0, 315.453],
                                               [0, 617.173, 242.259],
                                               [0, 0, 1]])
            self.__file_type = '_color.png'
        else:
            self.camera_intrinsics = np.array([])
            self.__file_type = '.png'

        super(SynthHandsDataLoader, self).__init__(name="SynthHands", data_path=data_path, subset=subset,
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

            file = '{}{}'.format(self.files[ij], self.__file_type)
            if load_img:
                #image = plt.imread(os.path.join(self.data_path, file))
                image = plt.imread(file)
                if np.max(image) <= 1:
                     image = (image * 255).astype(np.uint8)
                frames.append(image)
            # --- load joints coordinates ----
            coords.append( self._getCoordintanes('{}_joint_pos.txt'.format(self.files[ij])))
            # --------------------------
            paths.append(os.path.dirname(file).replace(os.path.join(self.data_path, ''), ''))
            names.append(os.path.basename(file).replace(self.__file_type, ''))

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

        files = [f for f in glob.iglob(os.path.join(self.data_path, subset)  + '**/**', recursive=True)
                 if re.search("\_joint_pos.txt", f)]
        List = [f.replace('_joint_pos.txt', '') for f in files]
        List.sort()

        return List

    def _getCoordintanes(self, file_path, n=3):
        with open(file_path) as f:
            contents = f.read()
            l = [float(idx) for idx in contents.split(",")]
            coords = np.array([l[i:i + n] for i in range(0, len(l), n)])

        skel_hom2d = np.array(self.camera_intrinsics).dot(coords.transpose()).transpose()
        skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

        return skel_proj
