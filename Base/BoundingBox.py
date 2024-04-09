from abc import ABC
import numpy as np
import os

from datasets.Base.dataconfig import BboxesSets
from common_utils import loadJSON, saveAsJSON

# =====================================================================================================================
class BoundingBox(ABC):

    def __init__(self, **kwargs):

        print(kwargs)

        self.dataset = kwargs['dataset']
        self.data_subset = kwargs['data_subset']
        self.bb_method = kwargs['bb_method']

        self._Bboxes = self._loadBbox()
        self.len = len(self._Bboxes)
        try:
            self.params = BboxesSets[self.dataset][self.bb_method]['params']
        except:
            self.params = []

        
    def _loadBbox(self):
        try:
            bboxes_path = BboxesSets[self.dataset][self.bb_method]['path']
        except:
            bboxes_path = ''

        bboxes_path = bboxes_path % self.data_subset if '%s' in bboxes_path else bboxes_path
        Bboxes = loadJSON(bboxes_path) if bboxes_path != '' else {}

        return Bboxes

            
    def add2Bbox(self, key, name, val):
        if key in self._Bboxes:
            if name in self._Bboxes[key]:
                self._Bboxes[key][name].append(val)
            else:
                self._Bboxes[key].update({name: val})
        else:
            self._Bboxes[key] = {name: val}
            
    
    def getBbox(self, key, name):
        if not (key in self._Bboxes and name in self._Bboxes[key]):
            return None
    
        if not len(self._Bboxes[key][name]):
            return None

        cls = []
        confidence = []
        if len(self._Bboxes[key][name][0]) > 4:
            cls = np.asarray(self._Bboxes[key][name])[:, 0]
            confidence = np.asarray(self._Bboxes[key][name])[:, 1]
            Bboxes = np.asarray(self._Bboxes[key][name])[:, 2:].astype(int)
        else:
            Bboxes = np.asarray(self._Bboxes[key][name]).astype(int)
    
        return Bboxes, cls, confidence

    def saveBbox(self, save_to):
        saveAsJSON(self._Bboxes, os.path.join(save_to, "bboxCoords.data"))
