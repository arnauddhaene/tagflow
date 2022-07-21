from typing import Tuple
from numpy.typing import ArrayLike
import itertools

import numpy as np
import torch
from torch import nn

from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter

import streamlit as st

from .models.tracking.resnet2 import ResNet2


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def unpack_circle(circle: ArrayLike) -> Tuple[float, float, float]:
    # Handles type hinting to return Cx, Cy, and R
    circle = np.array(circle)
    return circle[0], circle[1], circle[2]


def get_patch_path(ims, path, is_scaled=False, width=32):
    
    rad = width // 2
    
    if path.ndim == 1:
        path = path[:, None]

    if not is_scaled:
        p_path = (path + 0.5)
        p_path[1] *= ims.shape[-2]
        p_path[0] *= ims.shape[-1]
    else:
        p_path = path
        
    im_cp = np.pad(ims, pad_width=((0, 0), (rad + 1, rad + 1), (rad + 1, rad + 1)), mode='constant')

    pos1 = p_path[1, 0]
    ipos1 = int(pos1)

    pos0 = p_path[0, 0]
    ipos0 = int(pos0)

    im_c = im_cp[:, ipos1:ipos1 + 2 * rad + 2, ipos0:ipos0 + 2 * rad + 2]

    kim_c = np.fft.ifftshift(
        np.fft.fftn(np.fft.fftshift(im_c, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))

    rr = 2 * np.pi * np.arange(-(rad + 1), rad + 1) / width
    yy, xx = np.meshgrid(rr, rr, indexing='ij')

    kim_c *= np.exp(1j * xx[np.newaxis, ...] * (pos0 - ipos0))
    kim_c *= np.exp(1j * yy[np.newaxis, ...] * (pos1 - ipos1))

    im_c2 = np.abs(np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftshift(kim_c, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)))
    im_c2 = im_c2[:, 1:-1, 1:-1]
    
    c_path = path - path[:, 0][:, np.newaxis]
    
    return im_c2, c_path


@st.cache
def load_model(
    model_path: str,
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


def polar_to_cartesian(rho: float, theta: float) -> Tuple[float, float]:
    """Convert polar to cartesian coordinates

    Args:
        rho (float): Length of radius from center in pixels.
        theta (float): Angle in radian.

    Returns:
        Tuple[float, float]: (x, y) cartesian coordinates.
    """
    return rho * np.cos(theta), rho * np.sin(theta)


def generate_reference(
    radii: Tuple[float, float],
    circ: int = 24, radial: int = 7,
) -> np.ndarray:
    rho_m, rho_M = radii
    rhos = np.linspace(rho_m, rho_M, radial)
    thetas = np.linspace(-np.pi + (np.pi / circ), np.pi - (np.pi / circ), circ)
    
    polar_coords = np.array(list(itertools.product(rhos, thetas)))
    r, t = polar_coords[:, 0], polar_coords[:, 1]
    
    return np.vstack([r * np.cos(t), r * np.sin(t)]).T
