import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from .widgets.player import Player


@st.cache
def load_data():
    npz = np.load('torch_track/sample_data/in_vivo_data.npz')
    y1r = np.load('torch_track/sample_data/in_vivo_data_deformation.npz')['y1r']
    imt, r0 = npz['imt'], npz['r0']
    return imt, r0, y1r.swapaxes(0, 2)


def write():
    
    imt, r0, y1r = load_data()
    
    st.session_state.image = imt
    # st.session_state.points = y1r
    # st.session_state.reference = r0
        
    Player(st.session_state.image, st.session_state.points).display()
