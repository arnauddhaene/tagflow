import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from .base import BaseWidget
from ..state.state import SessionState, SessionStatus


class Player(BaseWidget):
    """Video player for displaying time-wise tracking and cardiac motion

    Attributes:
        Nt (int): time dimension
        speed (int): period is 1 / exp(speed)
        window (str): 'zoomed' or 'wide' view of the image
    """
    
    def __init__(self, aspect: float = .6):
        """Constructor"""
        
        ss = SessionState()
        
        if ss.status().value < SessionStatus.image.value:
            raise ValueError(f'Session must contain image to display {self.__class__.__name__}')

        self.aspect = aspect

        self.image: np.ndarray = ss.image.value()
        self.points: Optional[np.ndarray] = ss.deformation.value()
        self.Nt: int = self.image.shape[0]
        
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
        
        self.speed = st.sidebar.number_input('Speed', 0, 10, 5, 1)
        self.update_view()
        
        image = plt.imshow(self.image[st.session_state.frame], cmap='gray')
        if self.points is not None:
            paths = plt.scatter(self.points[st.session_state.frame, :, 0],
                                self.points[st.session_state.frame, :, 1],
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
                paths.set_offsets(self.points[st.session_state.frame])

            player.pyplot(plt)
        
    def update_view(self):
        """Update pyplot view to zoomed (either defined or padded by points) or wide"""

        xdim, ydim = tuple(self.image.shape[1:])
        shortest_dim = min(xdim, ydim)
        
        cx, cy = xdim / 2, ydim / 2

        xmin, xmax = \
            int(cx - (shortest_dim / (3 * self.aspect))), int(cx + (shortest_dim / (3 * self.aspect)))
        ymin, ymax = int(cy - (shortest_dim / 3)), int(cy + (shortest_dim / 3))
    
        plt.xlim(xmin, xmax)
        plt.ylim(ymax, ymin)
                
    def toggle_play(self):
        st.session_state.playing = not st.session_state.playing
        st.session_state.frame -= 1
    
    def reset(self):
        st.session_state.playing = False
        st.session_state.frame = 0
