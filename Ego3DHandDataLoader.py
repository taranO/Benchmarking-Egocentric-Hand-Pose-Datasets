import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

from datasets.Base.DataLoader import DataLoader

# ==========================================================================================================
class Ego3DHandDataLoader(DataLoader):

    def __init__(self, data_path, subset="", batch_size=1, shuffle=True, type='', IMG_H = 540, IMG_W = 960):

        self.data_path = os.path.join(data_path, "data")
        super(Ego3DHandDataLoader, self).__init__(name="Ego3DHands", data_path=data_path, subset=subset,
                                            batch_size=batch_size, shuffle=shuffle)

        self.__IMG_H = IMG_H
        self.__IMG_W = IMG_W
        self.__camIntrinsics()

        if type == 'depth':
            self.__file_type = 'depth.png'
        elif type == 'color':
            self.__file_type = 'color.png'
        else:
            self.__file_type = 'color_new.png'

    def __camIntrinsics(self):

        px = self.__IMG_H / 2
        py = self.__IMG_W / 2
        fx = 187.932 * self.__IMG_H / 270
        fy = 187.932 * self.__IMG_W / 480

        self.camera_intrinsics = np.array([[fx, 0, px],
                                           [0, fy, py],
                                           [0, 0, 1]])

    def getBatch(self, ind, load_img=True):
        frames = []
        coords = []
        paths  = []
        names  = []
        for ij in self._batches[ind]:
            # --- load image ----

            if self.files[ij] == "":
                continue

            file = '{}/{}'.format(self.files[ij], self.__file_type)

            if load_img:
                try:
                     #image = plt.imread(os.path.join(self.data_path, file))
                     image = plt.imread(file)
                except:
                     continue

                if np.max(image) <= 1:
                     image = (image * 255).astype(np.uint8)
                frames.append(image)
            # --- load joints coordinates ----
            coords.append( self._getCoordintanes('{}/location_2d.npy'.format(self.files[ij])))
            # --------------------------
            paths.append(os.path.dirname(file).replace(os.path.join(self.data_path, ''), ''))
            names.append(self.__file_type)

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

        files = [f for f in glob.iglob(os.path.join(self.data_path, subset) + '**/**', recursive=True) if re.search("\_2d.npy", f)]
        files = [f for f in files if os.path.getsize(f) > 0]
        List = [f.replace('/location_2d.npy', '') for f in files]
        List.sort()

        return List

    def _getCoordintanes(self, file_path):

        coords = np.load(file_path)[:,1:,:]
        coords[..., 0] *= self.__IMG_H
        coords[..., 1] *= self.__IMG_W

        return coords[..., [1, 0]]
