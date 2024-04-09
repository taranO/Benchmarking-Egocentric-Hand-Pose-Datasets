import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from datasets.Base.DataLoader import DataLoader

# ==========================================================================================================
class GanHandDataLoader(DataLoader):

    def __init__(self, data_path, subset="", batch_size=1, shuffle=True):

        self.data_path = os.path.join(data_path, 'data')

        super(GanHandDataLoader, self).__init__(name="GanHand", data_path=data_path, subset=subset,
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
                image = plt.imread(self.files[ij])
                if np.max(image) <= 1:
                    image *= 255
                image = image.astype(np.uint8)

                frames.append(image)
            # --- load joints coordinates ----
            path = os.path.dirname(self.files[ij]).replace(os.path.join(self.data_path, ""), "")
            name = os.path.basename(self.files[ij]).split('_')[0]
            coords.append(self._getCoordintanes(os.path.join(self.data_path, path, name + "_joint2D.txt")))
            # --------------------------
            paths.append(path)
            names.append(name)

        return frames, coords, paths, names

    def _getCoordintanes(self, file_path, n=2):
        coords = []

        with open(file_path) as f:
            contents = f.read()
            l = [float(idx) for idx in contents.split(",")]
            coords = [l[i:i + n] for i in range(0, len(l), n)]

        return coords

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
            frames = glob.glob(os.path.join(ss, "*.png"))
            frames.sort()
            List.extend(frames)
        return List
