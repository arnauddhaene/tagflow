import numpy as np
import pydicom

import streamlit as st

from .widgets.player import Player


@st.cache
def load_data():
    npz = np.load('sample_data/in_vivo_data.npz')
    y1r = np.load('sample_data/in_vivo_data_deformation.npz')['y1r']
    imt, r0 = npz['imt'], npz['r0']
    return imt, r0, y1r.swapaxes(0, 2)


def load_sample():
    imt, r0, _ = load_data()
    st.session_state.image = imt
    st.session_state.reference = r0


@st.cache
def init():
    """Instantiate session storage with application-wide variables"""
    storage = ['points', 'reference', 'roi', 'image']
    
    for item in storage:
        if item not in st.session_state:
            st.session_state[item] = None
            

def clear():
    st.legacy_caching.clear_cache()
    st.session_state.image = None
    st.session_state.roi = None
    st.session_state.reference = None


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
