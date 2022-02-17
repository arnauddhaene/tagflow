import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pydicom

import streamlit as st

from .widgets.player import Player


# @st.cache
# def load_data():
#     npz = np.load('torch_track/sample_data/in_vivo_data.npz')
#     y1r = np.load('torch_track/sample_data/in_vivo_data_deformation.npz')['y1r']
#     imt, r0 = npz['imt'], npz['r0']
#     return imt, r0, y1r.swapaxes(0, 2)


def write():
    
    # imt, _, _ = load_data()
    # st.session_state.image = imt
    # Player(st.session_state.image, st.session_state.points).display()
    
    if st.session_state.image is None:
        
        images = st.file_uploader('Upload series of DICOM files (in order)', type='dcm',
                                  accept_multiple_files=True)
        
        if len(images) > 0:
            res = map(lambda ds: (ds.InstanceNumber, ds.pixel_array), map(pydicom.dcmread, images))
            
            st.session_state.image = np.array(list(zip(*sorted(res, key=lambda item: item[0])))[1])
                        
            Player(st.session_state.image, st.session_state.points).display()
    else:
        Player(st.session_state.image, st.session_state.points).display()
