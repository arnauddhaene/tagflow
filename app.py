import itertools
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

from flask import Flask, request, jsonify

import cv2
import numpy as np
import torch
from torch import nn

from tagsim.utils import get_patch_path
from torch_track.network_resnet2 import ResNet2

MODEL_PATH = Path('torch_track/network_saves/resnet2_grid_tracking.pt')

app = Flask(__name__)


@app.route('/')
def root():
    return 'Hi there'


@app.route('/track', methods=['POST'])
def predict():

    if request.method == 'POST':

        payload = request.get_json(force=True)
        imt = np.array(payload['images'])
        r0 = np.array(payload['points'])

        # Get patches for each point in r0
        # Number of tracking points
        N = r0.shape[0]
        # Create empty array of Time x Height x Width for each tracking point
        X = np.empty((N, 1, 25, 32, 32), np.float32)

        for i, point in enumerate(r0):
            im_p, _ = get_patch_path(imt, point, is_scaled=True)
            X[i] = im_p.copy()

        X = X - (X.mean(axis=0) / X.std(axis=0))

        batch_size = 8
        N_batches = int(np.ceil(N / batch_size))

        device = torch.device('cpu')
        model = load_model(device=device)

        y1 = []

        with torch.no_grad():
            for i in tqdm(range(N_batches)):
                x = X[i * batch_size:(i + 1) * batch_size]
                x = torch.from_numpy(x).to(device)
                y_pred = model(x)
                y1.append(y_pred.detach().cpu().numpy())

        y1 = np.vstack(y1)
        y1 = y1.reshape(-1, 2, 25)

        result = {
            'prediction': y1.tolist()
        }

        return jsonify(result)


@app.route('/hough', methods=['POST'])
def get_roi():

    if request.method == 'POST':

        payload = request.get_json(force=True)
        hc_input = np.array(payload['image'])

        params = ['dp', 'min_d', 'min_r', 'max_r', 'p1', 'p2', 'circ', 'radial']
        dp, min_d, min_r, max_r, p1, p2, circ, radial = tuple(map(lambda k: payload[k],
                                                                  params))

        circles = cv2.HoughCircles(hc_input.astype(np.uint8),
                                   cv2.HOUGH_GRADIENT,
                                   dp=dp, minDist=min_d, param1=p1, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)

        if circles is not None:
            circles = np.uint16(np.around(circles))
        else:
            # If no circles found, go for center
            circles = np.array(
                [[[hc_input.shape[0] / 2, hc_input.shape[1] / 2, 25]]])

        Cx, Cy, R = tuple(circles[0, 0])

        # Same concept as Ferdian's paper - 128 totalx/ tracking points
        rhos = np.linspace(R * 0.5, R * 0.9, radial)
        thetas = np.linspace(-np.pi + (np.pi / circ), np.pi - (np.pi / circ), circ)

        polar_coords = np.array(list(itertools.product(rhos, thetas)))

        r0 = np.array(list(map(
            lambda pt: polar_to_cartesian(*pt) + np.array([Cx, Cy]),
            polar_coords)))

        result = {
            'points': r0.tolist(),
            'roi': circles[0, 0].tolist()
        }

        return jsonify(result)
    

def polar_to_cartesian(rho: float, theta: float) -> Tuple[float, float]:
    """Convert polar to cartesian coordinates

    Args:
        rho (float): Length of radius from center in pixels.
        theta (float): Angle in radian.

    Returns:
        Tuple[float, float]: (x, y) cartesian coordinates.
    """
    return rho * np.cos(theta), rho * np.sin(theta)


def load_model(
    model_path: str = MODEL_PATH,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """Load saved model

    Args:
        model_path (str, optional): Saved model. Defaults to MODEL_PATH.
        device (torch.device, optional): Device to store model on.
            Defaults to torch.device('cpu').

    Returns:
        torch.Module: model stored on device.
    """

    kwargs = dict(do_coordconv=True, fc_shortcut=False)
    model = ResNet2([2, 2, 2, 2], **kwargs)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    return model.to(device)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
