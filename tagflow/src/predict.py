from numpy.typing import ArrayLike

from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch

from ..utils import load_model, get_patch_path

MODEL_PATH = Path(__file__).parent.parent / 'network_saves/resnet2_grid_tracking.pt'


def predict(imt: ArrayLike, r0: ArrayLike) -> ArrayLike:
    
    # Number of reference tracking points
    N = r0.shape[0]
    
    # Empty T x W x H for each reference tracking point
    X = np.empty((N, 1, 25, 32, 32), np.float32)
    
    for i, point in enumerate(r0):
        im_p, _ = get_patch_path(imt, point, is_scaled=True)
        X[i] = im_p.copy()

    X = X - (X.mean(axis=0) / X.std(axis=0))

    batch_size = 8
    N_batches = int(np.ceil(N / batch_size))

    device = torch.device('cpu')
    model = load_model(MODEL_PATH, device=device)

    y1 = []

    with torch.no_grad():
        for i in tqdm(range(N_batches)):
            x = X[i * batch_size:(i + 1) * batch_size]
            x = torch.from_numpy(x).to(device)
            y_pred = model(x)
            y1.append(y_pred.detach().cpu().numpy())

    y1 = np.vstack(y1)
    y1 = y1.reshape(-1, 2, 25)
    
    return y1 + r0[:, :, None]
