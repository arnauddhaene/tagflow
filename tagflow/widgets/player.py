import time

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

import streamlit as st

from .base import BaseWidget


class Player(BaseWidget):
    """Video player for displaying time-wise tracking and cardiac motion

    Attributes:
        image (ArrayLike): the (T x W x H) input image
        points (ArrayLike): the (T x 2 x Npoints) tracked points
        Nt (int): time dimension
        speed (int): period is 1 / exp(speed)
        window (str): 'zoomed' or 'wide' view of the image
    """
    
    def __init__(self, image: ArrayLike, points: ArrayLike = None):
        """Constructor

        Args:
            image (ArrayLike): the (T x W x H) input image
            points (ArrayLike): Tracked points (time x 2 x Npoints). Defaults to None.
        """
        self.image = image
        self.points = points
        self.Nt = image.shape[0]
        
        self.init_state()
        
    @st.cache
    def init_state(self):
        """Instanciate necessary sessions state key-value pairs"""
        if 'frame' not in st.session_state:
            st.session_state.frame = 0
        if 'playing' not in st.session_state:
            st.session_state.playing = False
            
    def display(self):
        """Display player by updating pyplot every 1 / exp(speed)"""
        col1, col2 = st.sidebar.columns(2)
        
        self.speed = col1.number_input('Speed', 0, 10, 5, 1)
        self.window = col2.selectbox('View', ['zoomed', 'wide'])
        self.update_view()
        
        image = plt.imshow(self.image[st.session_state.frame], cmap='gray')
        if self.points is not None:
            paths = plt.scatter(self.points[st.session_state.frame, 0, :],
                                self.points[st.session_state.frame, 1, :],
                                15, c='r', marker='x')

        plt.axis('off')
        player = st.pyplot(plt)
        
        left, lcenter, center, rcenter, right = st.columns(5)
        lcenter.button('⏮️', on_click=self.reset)
        center.button('⏯', on_click=self.toggle_play)

        while st.session_state.playing:
            st.session_state.frame = (st.session_state.frame + 1) % self.Nt
            time.sleep(1 / np.exp(self.speed))
            image.set_data(self.image[st.session_state.frame])
            if self.points is not None:
                paths.set_offsets(self.points[st.session_state.frame].swapaxes(0, 1))

            player.pyplot(plt)
        
    def update_view(self):
        """Update pyplot view to zoomed (either defined or padded by points) or wide"""
        if self.window == 'zoomed':
            height, width = tuple(self.image.shape[1:])
                
            if self.points is None:
                xmin, xmax = width * .25, width * .75
                ymin, ymax = height * .25, height * .75
            else:
                xmin, xmax = self.points[:, 0, :].min(), self.points[:, 0, :].max()
                ymin, ymax = self.points[:, 1, :].min(), self.points[:, 1, :].max()
            
            plt.xlim(max(0, xmin - 20), min(xmax + 20, width))
            plt.ylim(min(ymax + 20, height), max(0, ymin - 20))
                
    def toggle_play(self):
        st.session_state.playing = not st.session_state.playing
        st.session_state.frame -= 1
    
    def reset(self):
        st.session_state.playing = False
        st.session_state.frame = 0
