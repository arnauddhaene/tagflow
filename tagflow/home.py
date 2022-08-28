import functools
import operator
import itertools

import numpy as np
from scipy import ndimage

import h5py

import streamlit as st

from .widgets.player import Player
from .state.state import SessionState, SessionStatus
from .widgets.reference_picker import ReferencePicker


@st.cache
def load_data():
    npz = np.load('sample_data/in_vivo_data.npz')
    y1r = np.load('sample_data/in_vivo_data_deformation.npz')['y1r']
    imt, r0 = npz['imt'], npz['r0']
    return imt, r0, np.rollaxis(y1r, 2)


def load_sample():
    imt, r0, y1r = load_data()
    
    centre = r0.mean(axis=0).T
    radius = 1.1 * np.abs(np.linalg.norm(centre - r0, axis=1)).max()
    
    circle = np.array([*centre, radius])
    shape = imt.shape[1:]
    roi = ReferencePicker.circle_mask(circle, shape, 0.9) ^ \
        ReferencePicker.circle_mask(circle, shape, 0.5)
            
    ss = SessionState()

    ss.roi.update(roi)
    ss.image.update(imt)
    ss.reference.update(r0)
    ss.deformation.update(y1r)
    

@st.cache
def init():
    SessionState()
            

def clear():
    SessionState().clear()


def write():

    st.write("""
        # Home
    """)
    
    ss = SessionState()
        
    if ss.status().value < SessionStatus.image.value:
        st.sidebar.button('Use sample image', on_click=load_sample)

        st.sidebar.write("""
            Conversion from DICOM or Nifti can be done using
            [this script](https://gist.github.com/arnauddhaene/d85af1b923881e42ee4a73bdda4b2487).
        """)
        shape = st.sidebar.text_input('Array shape (e.g., WHT, HTW)', value='TWH')
        datafile = st.sidebar.file_uploader('Upload sequence of images in HDF5 format', type='h5',
                                            accept_multiple_files=False)

        # Check that text input is legal
        if shape in map(functools.partial(functools.reduce, operator.add), itertools.permutations('TWH', 3)):
        
            if datafile:
                # Get data from file
                hf = h5py.File(datafile, 'r')
                dataset = st.sidebar.selectbox('Choose dataset', hf.keys())
                data_array = np.array(hf.get(dataset))
                # Fix axes order and interpolate in time to fit to 25
                data_array = np.moveaxis(data_array, tuple(map(shape.index, ['T', 'W', 'H'])), (0, 1, 2))
                if data_array.shape[0] != 25:
                    data_array = ndimage.zoom(data_array, (25. / data_array.shape[0], 1., 1.))

                ss.image.update(data_array)
                
                Player().display()
        else:
            st.warning(f'Given shape: {shape} is not valid. Please follow convention.')
            
    else:
        st.sidebar.button('Clear current image', on_click=clear)
        if ss.status().value < SessionStatus.reference.value:
            aspect = st.number_input('Padding from reference center', .5, 1.5, .6, .1)
        else:
            aspect = .6
        Player(aspect).display()
