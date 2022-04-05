import streamlit as st

from .widgets.hough_reference import HoughReference
from .widgets.nn_reference import NeuralReference
from .state.state import SessionState


def write():
    ss = SessionState()

    if ss.image.value() is not None:
        
        annot = st.sidebar.selectbox('Annotation', ['Neural Network', 'Hough Transform'])
        
        if annot == 'Hough Transform':
            HoughReference().display()
        elif annot == 'Neural Network':
            NeuralReference().display()
        
    else:
        st.warning('Please uploade an image to use the Hough segmentation algorithm.')
