import requests
from pathlib import Path

import numpy as np


TRACK_MODEL_PATH = Path(__file__).parent.parent / 'network_saves/resnet2_grid_tracking.pt'
# URL_API = 'http://172.17.0.5:5000'
URL_API = 'http://127.0.0.1:5000'


def track(imt: np.ndarray, r0: np.ndarray) -> np.ndarray:
    
    payload = {'images': imt.tolist(), 'points': r0.tolist()}
    req = requests.post(URL_API + '/track', json=payload)
    
    return np.array(req.json()['prediction'])


def segment(imt: np.ndarray) -> np.ndarray:

    payload = {'image': imt.tolist()}
    req = requests.post(URL_API + '/segment', json=payload)
    
    return np.array(req.json()['prediction'])
