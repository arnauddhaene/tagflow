from numpy.typing import ArrayLike

import cv2
import numpy as np

import streamlit as st

from .reference_picker import ReferencePicker
from ..src.hough import hough_circle


class HoughReference(ReferencePicker):
    """The Hough Transform reference tracking point picker

    Attributes:
        image (ArrayLike): the (T x W x H) input image
        ref_points (ArrayLike): the (Npoints x 2) reference tracking points
        roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
        kernel (int): Kernel size of blurring filter for preprocessing
    """
    
    def __init__(self, kernel: int = 5):
        """Constructor d

        Args:
            kernel (int, optional): Blurring kernel size. Defaults to 5.
        """
        super().__init__()

        self.kernel: int = kernel
                
    def preprocess(self) -> ArrayLike:
        """Preprocessing pipeline before applying the Hough Transform

        Returns:
            ArrayLike: NumPy array of dimensions (w x h)
        """
        # hc_input = self.image.var(axis=0) / self.image.mean(axis=0)
        hc_input = 255 - (self.image[4] - self.image.mean(axis=0))
        hc_input = cv2.filter2D(hc_input, -1,
                                np.ones((self.kernel, self.kernel), np.float32) / self.kernel ** 2)
        
        return hc_input
    
    def reference(self):
        """Computes reference tracking points
        
        Modifies:
            ref_points (ArrayLike): reference tracking points (Npoints x 2)
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
        """

        col1, col2 = st.sidebar.columns(2)

        min_d = col1.slider('Minimum distance', 2., 200., 200.)
        dp = col2.slider('DP', .9, 2., 1.)
        min_r = col1.slider('Minimum radius', 1, 50, 10)
        max_r = col2.slider('Maximum radius', 10, 100, 30)
        p1 = col1.slider('Parameter 1', 10., 200., 70.)
        p2 = col2.slider('Parameter 2', .5, 1., .8)
        
        circ = col1.number_input('Circumferential grid', 4, 30, 6, 1, key='circ')
        radial = col2.number_input('Radial grid', 3, 10, 4, 1, key='radial')
        
        hc_input = self.preprocess()
                
        self.ref_points, circle = hough_circle(hc_input,
                                               dp, min_d, p1, p2, min_r, max_r,
                                               int(circ), int(radial))
                
        self.roi = self.circle_mask(circle, hc_input.shape, 0.9) ^ \
            self.circle_mask(circle, hc_input.shape, 0.5)
