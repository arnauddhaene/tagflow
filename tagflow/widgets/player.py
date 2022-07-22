from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

import streamlit.components.v1 as components

from .base import BaseWidget
from ..state.state import SessionState, SessionStatus


class Player(BaseWidget):
    """Video player for displaying time-wise tracking and cardiac motion

    Attributes:
        Nt (int): time dimension
        speed (int): period is 1 / exp(speed)
        window (str): 'zoomed' or 'wide' view of the image
    """
    
    def __init__(self, room: float = 0.5):
        """Constructor"""
        
        ss = SessionState()
        
        if ss.status().value < SessionStatus.image.value:
            raise ValueError(f'Session must contain image to display {self.__class__.__name__}')

        self.room = room

        self.image: np.ndarray = ss.image.value()
        self.points: Optional[np.ndarray] = ss.deformation.value()
    
    def display(self):
        """Display player by updating pyplot every 1 / exp(speed)"""
        
        va = VideoAnimation(self.image, self.points, self.room)
        components.html(va.anim.to_jshtml(), height=1000)


class VideoAnimation:

    def __init__(self, imt: np.ndarray, defo: np.ndarray = None, room: float = 0.5):

        self.imt = imt
        self.pts = defo
        self.room = room

        self.fig, self.axarr = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.im = self.axarr[0, 0].imshow(self.imt[0], cmap="gray")
        if self.pts is not None:
            self.pt, = self.axarr[0, 0].plot(*self.pts[0].T, linestyle='None', c='r', marker='x')

            # Find center of deformation
            cx, cy = self.pts.mean(axis=(0, 1))
            padding = (np.min(self.imt.shape[1:]) / 2) * self.room
            self.axarr[0, 0].set_xlim(cx - padding, cx + padding)
            self.axarr[0, 0].set_ylim(cy + padding, cy - padding)

        self.axarr[0, 0].axis('off')

        self.anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_animation,
            frames=imt.shape[0],
            interval=50,
            blit=True
        )

        plt.close()

    def init_animation(self):
        self.im.set_data(self.imt[0])
        if self.pts is not None:
            self.pt.set_data(*self.pts[0].T)
            return [self.im, self.pt]
        else:
            return [self.im, ]

    def animate(self, i):
        self.im.set_data(self.imt[i])
        if self.pts is not None:
            self.pt.set_data(*self.pts[i].T)
            return [self.im, self.pt]
        else:
            return [self.im, ]
