import numpy as np
import pydicom

import streamlit as st

from .widgets.player import Player
from .state.state import SessionState


@st.cache
def load_data():
    npz = np.load('sample_data/in_vivo_data.npz')
    y1r = np.load('sample_data/in_vivo_data_deformation.npz')['y1r']
    imt, r0 = npz['imt'], npz['r0']
    return imt, r0, y1r


def load_sample():
    imt, r0, y1r = load_data()
    ss = SessionState()

    ss.image.update(imt)
    ss.reference.update(r0)
    ss.deformation.update(y1r.swapaxes(0, 2))
    
    # centre = r0.mean(axis=0).T
    # radius = 1.1 * np.abs(np.linalg.norm(centre - r0, axis=1)).max()

    # st.session_state.roi = np.array([*centre, radius])


@st.cache
def init():
    SessionState()
            

def clear():
    SessionState().clear()


def write():
        
    if st.session_state.image is None:
        
        st.sidebar.button('Use sample image', on_click=load_sample)
        
        images = st.sidebar.file_uploader('Upload series of DICOM files (in order)', type='dcm',
                                          accept_multiple_files=True)
        
        if len(images) > 0:
            res = map(lambda ds: (ds.InstanceNumber, ds.pixel_array), map(pydicom.dcmread, images))
            
            st.session_state.image = np.array(list(zip(*sorted(res, key=lambda item: item[0])))[1])
                        
            Player(st.session_state.image, st.session_state.points).display()

    else:
        st.sidebar.button('Clear current image', on_click=clear)
                        
        Player(st.session_state.image, st.session_state.points).display()
