import sys
import os

from pathlib import Path
pth = Path.home()
# ===========================================================================================================================

POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8],
              [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16],
              [0, 17], [17, 18], [18, 19], [19, 20]]

Thr = 35
step = 5
nPoints = 21

BboxesSets = {
    "H2O": {
        "": {
            "path": "",
        },
        "yolov2": {
            "path" : "<replace with correct path>/darknet/H2O/%s/bboxCoords.data",
        },
        "media_pipe": {
            "path": os.path.join(pth, "<replace with correct path>/MediaPipe/H2O/%s/bboxCoords.data"),
            "params": {
                "max_num_hands": 2,
                "static_image_mode": True,
                "detection_confidence": 0.1,
                "padding": 50
            }
        },
        "rtmdet": {
            "path": "<replace with correct path>/mmpose/H2O/%s/rtmdet/bboxCoords.data"
        },
        "yolo": {
            "path": "<replace with correct path>/mmpose/H2O/%s/yolo/bboxCoords.data"
        },
        "yolov4-tiny": {
            "path": "<replace with correct path>/mmpose/H2O/%s/yolov4-tiny/bboxCoords.data"
        },
        "tiny": {
            "path": "<replace with correct path>/mmpose/H2O/%s/tiny/bboxCoords.data"
        },
        "tiny-prn": {
            "path": "<replace with correct path>/mmpose/H2O/%s/tiny-prn/bboxCoords.data"
        }
    },
    "FPHA": {
        "": {
            "path": "",
        },
        "yolov2": {
            "path": "<replace with correct path>/darknet/FPHA/%s/bboxCoords.data",
        },
        "media_pipe": {
            "path": "<replace with correct path>/MediaPipe/FPHA/%s/bboxCoords.data",
            "params": {
                "max_num_hands": 2,
                "static_image_mode": True,
                "detection_confidence": 0.1,
                "padding": 50
            }
        },
    },
    "GANeratedHands": {
        "": {
            "path": "",
        },
        "media_pipe": {
            "path": "",
            "params": {
                "max_num_hands": 1,
                "static_image_mode": True,
                "detection_confidence": 0.1,
            }
        },
    },
    "SynthHands": {
        "": {
            "path": "",
        },
        "media_pipe": {
            "path": "",
            "params": {
                "max_num_hands": 1,
                "static_image_mode": True,
                "detection_confidence": 0.1,
            }
        },
    },
    "Ego3DHands": {
        "": {
            "path": "",
        },
        "yolov2": {
            "path": "<replace with correct path>/darknet/Ego3DHands/%s/bboxCoords.data",
        },
        "media_pipe": {
            "path": "<replace with correct path>/MediaPipe/Ego3DHands/%s/bboxCoords.data",
            "params": {
                "max_num_hands": 2,
                "static_image_mode": True,
                "detection_confidence": 0.1,
                "padding": 50
            }
        },
        "rtmdet": {
            "path": "<replace with correct path>/mmpose/Ego3DHands/%s/rtmdet/bboxCoords.data"
        },
        "yolo": {
            "path": "<replace with correct path>/mmpose/Ego3DHands/%s/yolo/bboxCoords.data"
        },
        "yolov4-tiny": {
            "path": "<replace with correct path>/mmpose/Ego3DHands/%s/yolov4-tiny/bboxCoords.data"
        },
        "tiny": {
            "path": "<replace with correct path>/mmpose/Ego3DHands/%s/tiny/bboxCoords.data"
        },
        "tiny-prn": {
            "path": "<replace with correct path>/mmpose/Ego3DHands/%s/tiny-prn/bboxCoords.data"
        }
    },
    "HoloLens": {
        "": {
            "path": "",
        },
        "yolov2": {
            "path": "<replace with correct path>/darknet/HoloLens/%s/bboxCoords.data",
        },
        "media_pipe": {
            "path": "<replace with correct path>/MediaPipe/HoloLens/%s/bboxCoords.data",
            "params": {
                "max_num_hands": 2,
                "static_image_mode": True,
                "detection_confidence": 0.1,
                "padding": 50
            }
        },
    },
    "GoPro": {
        "": {
            "path": "",
        },
        "yolov2": {
            "path": "", #"<replace with correct path>/darknet/GoPro/%s/bboxCoords.data",
        },
        "media_pipe": {
            "path": "", # "<replace with correct path>/MediaPipe/GoPro/%s/bboxCoords.data",
            "params": {
                "max_num_hands": 2,
                "static_image_mode": True,
                "detection_confidence": 0.1,
                "padding": 50
            }
        },
    },
}


# ===========================================================================================================================
def getDataLoader(dataset, dataset_path, data_subset, batch_size=1):

    if dataset == "H2O":
        from datasets.H2O.src.H2ODataLoader import H2ODataLoader
        dataLoader = H2ODataLoader(dataset_path, subset=data_subset, batch_size=batch_size)

    elif dataset == "FPHA":
        from datasets.FPHA.src.FPHADataLoader import FPHADataLoader
        dataLoader = FPHADataLoader(dataset_path, subset=data_subset, batch_size=batch_size)

    elif dataset == "GANeratedHands":
        from datasets.GANeratedHands.src.GanHandDataLoader import GanHandDataLoader
        dataLoader = GanHandDataLoader(dataset_path, subset=data_subset, batch_size=batch_size)

    elif dataset == "SynthHands":
        from datasets.SynthHands.src.SynthHandsDataLoader import SynthHandsDataLoader
        dataLoader = SynthHandsDataLoader(dataset_path, subset=data_subset, batch_size=batch_size)

    elif dataset == "Ego3DHands":
        from datasets.Ego3DHands.src.Ego3DHandDataLoader import Ego3DHandDataLoader
        dataLoader = Ego3DHandDataLoader(dataset_path, subset=data_subset, batch_size=batch_size)

    return dataLoader
