import requests

import cv2
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from .widgets.hough_reference import HoughReference
from .widgets.manual_reference import ManualReference


def write():
    if st.session_state.image is not None:
        
        annot = st.sidebar.selectbox('Annotation', ['Hough Transform', 'Manual'])
        
        if annot == 'Hough Transform':
            HoughReference(st.session_state.image).display()
        elif annot == 'Manual':
            ManualReference(st.session_state.image).display()
        
    else:
        st.warning('Please uploade an image to use the Hough segmentation algorithm.')
