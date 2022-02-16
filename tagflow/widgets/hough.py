from numpy.typing import ArrayLike

import requests

import cv2
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from .base import BaseWidget


class Hough(BaseWidget):
    
    def __init__(self, image: ArrayLike, kernel: int = 5):
        
        self.image = image
        self.kernel = kernel
        
    def preprocess(self):
        
        # hc_input = self.image.var(axis=0) / self.image.mean(axis=0)
        hc_input = 255 - (self.image[4] - self.image.mean(axis=0))
        hc_input = cv2.filter2D(hc_input, -1,
                                np.ones((self.kernel, self.kernel), np.float32) / self.kernel ** 2)
        
        return hc_input
    
    def reference(self, hc_input: ArrayLike, dp: float = 1., min_d: float = 200.,
                  min_r: int = 10, max_r: int = 30, p1: float = 70., p2: float = .8):
        
        hc_params = dict(dp=dp, min_d=min_d, min_r=min_r, max_r=max_r, p1=p1, p2=p2)
        
        payload = {'image': hc_input.tolist(), **hc_params}
        hc_result = requests.post('http://127.0.0.1:5000/hough', json=payload).json()
        
        return np.array(hc_result['points']), np.array(hc_result['roi'])
        
    def display(self):

        hc_input = self.preprocess()

        col1, col2 = st.columns(2)

        dp = col1.slider('DP', .9, 2., 1.)
        min_d = col2.slider('Minimum distance between circles', 2., 200., 200.)
        min_r = col1.slider('Minimum radius', 1, 50, 10)
        max_r = col2.slider('Maximum radius', 10, 100, 30)
        p1 = col1.slider('Parameter 1', 10., 200., 70.)
        p2 = col2.slider('Parameter 2', .5, 1., .8)

        r0, circle = self.reference(hc_input, dp, min_d, min_r, max_r, p1, p2)

        # Plot the points we are tracking
        fig, ax = plt.subplots(1, figsize=(12, 8))

        ax.imshow(self.image[4], cmap='gray')
        ax.scatter(r0[:, 0], r0[:, 1], 30, c='r', marker='x')
        ax.axis('off')

        st.pyplot(fig)

        save, clear = st.columns(2)

        if save.button('Save reference tracking points'):
            st.session_state.reference = r0
            st.session_state.roi = circle
        
        if st.session_state.reference is not None:
            if clear.button('Clear reference tracking points'):
                st.session_state.reference = None
                st.session_state.points = None
                st.session_state.roi = None
