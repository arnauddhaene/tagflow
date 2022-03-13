from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from .base import BaseWidget


class ReferencePicker(BaseWidget):
    """Parent Widget class for a reference tracking point set picker

    Attributes:
        image (np.ndarray): the (T x W x H) input image
        ref_points (np.ndarray): the (Npoints x 2) reference tracking points
        roi (np.ndarray): circle coordinates for outer ROI [Cx, Cy, R]
    """
    
    def __init__(self, image: np.ndarray):
        """Constructor

        Args:
            image (ArrayLike): the (T x W x H) input image
        """
        self.image: np.ndarray = image
        self.ref_points: Optional[np.ndarray] = None
        self.roi: Optional[np.ndarray] = None
    
    def reference(self):
        """This method should compute self.ref_points and self.roi
        
        Modifies:
            ref_points (ArrayLike): reference tracking points (Npoints x 2)
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]

        Raises:
            NotImplementedError: forces children classes to implement this
        """
        raise NotImplementedError()
        
    def display(self):
        """Display in streamlit application"""

        self.reference()
        self.plot()
        
        save, clear = st.sidebar.columns(2)

        save.button('Save reference', on_click=self.save_reference)
            
        if st.session_state.reference is not None:
            clear.button('Clear reference', on_click=self.clear_reference)

    def plot(self):
        """Plots the input image's first timepoint and the compute reference tracking points."""
        
        # Plot the points we are tracking
        fig, ax = plt.subplots(1, figsize=(12, 8))

        ax.imshow(self.image[0], cmap='gray')
        if self.roi is not None:
            ax.imshow(self.roi, alpha=.3)
        if self.ref_points is not None:
            ax.scatter(self.ref_points[:, 0], self.ref_points[:, 1], 30, c='r', marker='x')
        ax.axis('off')
        
        self.zoom_in()
        
        st.pyplot(fig)

    def zoom_in(self):
        """Zooms in on the plotted image based on center or padded reference tracking points."""
        
        height, width = tuple(self.image.shape[1:])

        if self.ref_points is None:
            xmin, xmax = width * .25, width * .75
            ymin, ymax = height * .25, height * .75
        else:
            xmin, xmax = self.ref_points[:, 0].min(), self.ref_points[:, 0].max()
            ymin, ymax = self.ref_points[:, 1].min(), self.ref_points[:, 1].max()
        
        plt.xlim(max(0, xmin - 20), min(xmax + 20, width))
        plt.ylim(min(ymax + 20, height), max(0, ymin - 20))

    def save_reference(self):
        """Save reference tracking points to sessions state"""
        st.session_state.reference = self.ref_points
        st.session_state.roi = self.roi
        
    def clear_reference(self):
        """Clear reference tracking points from sessions state"""
        st.session_state.reference = None
        st.session_state.points = None
        st.session_state.roi = None
