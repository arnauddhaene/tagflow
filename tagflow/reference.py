import requests

import cv2
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from .widgets.hough import HoughReference


def write():
    if st.session_state.image is not None:
        HoughReference(st.session_state.image).display()
    else:
        st.warning('Please uploade an image to use the Hough segmentation algorithm.')
