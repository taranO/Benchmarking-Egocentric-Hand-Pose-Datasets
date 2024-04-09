from abc import ABC, abstractmethod
import numpy as np
import random

class DataLoader(ABC):

    def __init__(self, **kwargs):
        
        self.name = kwargs['name']
        self.main_path = kwargs['data_path']
        self.subset = kwargs['subset']
        self.batch_size = kwargs['batch_size']

        self.files = self._getFileList()
        if kwargs['shuffle']:
            random.shuffle(self.files)

        self._batches = self._chunk()
        self.n_batches = len(self._batches)


    def getBatchByIndx(self, ind):
        return self._batches[ind]

    @abstractmethod
    def getBatch(self):
        pass

    @abstractmethod
    def _getFileList(self):
        pass

    @abstractmethod
    def _getCoordintanes(self):
        pass

    def _chunk(self):
        indices = np.arange(len(self.files)).astype(int)
        return [indices[i * self.batch_size:(i + 1) * self.batch_size] for i in
                range((len(indices) + self.batch_size - 1) // self.batch_size)]

