import streamlit as st

from .widgets.hough_reference import HoughReference
from .widgets.manual_reference import ManualReference
from .widgets.nn_reference import NeuralReference


def write():
    if st.session_state.image is not None:
        
        annot = st.sidebar.selectbox('Annotation', ['Neural Network', 'Hough Transform', 'Manual'])
        
        if annot == 'Hough Transform':
            HoughReference(st.session_state.image).display()
        elif annot == 'Manual':
            ManualReference(st.session_state.image).display()
        elif annot == 'Neural Network':
            NeuralReference(st.session_state.image).display()
        
    else:
        st.warning('Please uploade an image to use the Hough segmentation algorithm.')
