import time

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

import streamlit as st

from .base import BaseWidget


class Player(BaseWidget):
    
    def __init__(self, image: ArrayLike, points: ArrayLike = None):
        """Constructor
        

        Args:
            image (ArrayLike): Image (time x width x height)
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
        """Display player"""
        col1, col2 = st.columns(2)
        
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
        if self.window == 'zoomed':
            if self.points is None:
                height, width = tuple(self.image.shape[1:])
                
                xmin, xmax = width * .25, width * .75
                ymin, ymax = height * .75, height * .25
            else:
                xmin, xmax = self.points[:, 0, :].min(), self.points[:, 0, :].max()
                ymin, ymax = self.points[:, 1, :].min(), self.points[:, 1, :].max()
            
            plt.xlim(xmin * .9, xmax * 1.05)
            plt.ylim(ymin * .9, ymax * 1.05)
                
    def toggle_play(self):
        st.session_state.playing = not st.session_state.playing
        st.session_state.frame -= 1
    
    def reset(self):
        st.session_state.playing = False
        st.session_state.frame = 0
