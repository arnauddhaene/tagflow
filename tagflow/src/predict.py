from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..utils import load_model, get_patch_path
from ..models.segmentation.unet import UNet
from ..models.tracking.resnet2 import ResNet2


TRACK_MODEL_PATH = Path(__file__).parent.parent / 'network_saves/resnet2_grid_tracking.pt'
ROI_MODEL_PATH = Path(__file__).parent.parent / 'network_saves/model_cine_tag_v1_sd.pt'


def track(imt: np.ndarray, r0: np.ndarray, in_st: bool = True, verbose: int = 0) -> np.ndarray:
    
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
    if in_st:
        model = load_model(TRACK_MODEL_PATH, device=device)
    else:
        kwargs = dict(do_coordconv=True, fc_shortcut=False)
        model = ResNet2([2, 2, 2, 2], **kwargs)
        model.load_state_dict(
            torch.load(TRACK_MODEL_PATH, map_location=device)
        )

        model = model.to(device)

    _y1 = []

    with torch.no_grad():
        for i in tqdm(range(N_batches), disable=(verbose < 1)):
            x = X[i * batch_size:(i + 1) * batch_size]
            x = torch.from_numpy(x).to(device)
            y_pred = model(x)
            _y1.append(y_pred.detach().cpu().numpy())

    y1: np.ndarray = np.vstack(_y1)
    y1 = y1.reshape(-1, 2, 25)
    
    return y1 + r0[:, :, None]


def segment(imt: torch.Tensor) -> torch.Tensor:

    if len(imt.shape) != 4:
        raise ValueError(f'Expects 4 dimensions: B, C, W, H. Got {imt.shape}')

    model: nn.Module = UNet(n_channels=1, n_classes=4, bilinear=True).double()
    # Load old saved version of the model as a state dictionary
    saved_model_sd: OrderedDict = torch.load(ROI_MODEL_PATH)
    # Extract UNet if saved model is parallelized
    model.load_state_dict(saved_model_sd)

    out: torch.Tensor = model(imt)
    return F.softmax(out, dim=1).argmax(dim=1).detach().numpy()
