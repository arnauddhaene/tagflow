from typing import Dict

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import streamlit as st
import streamlit.components.v1 as components

from .base import BaseWidget
from ..state.state import SessionState
from ..src.case import EvaluationCase


class StrainEstimator(BaseWidget):
    """Strain estimation widget

    Attributes:
        cx, cy, radius (float): coordinates of LV ROI
        mesh (ArrayLike): Set of all pixel coordinates within LV wall mask (2 x Npoints)
        Nt (int): timepoints
        Np (int): number of pixels in mesh
        points (ArrayLike): the (Npoints x 2 x time) tracked points
        rbf (RBF): the Radial Bias Function instance used for interpolation
        gl_strain (ArrayLike): the computed strain (time x dir x point) with dirs circumferential,
            radial, and longitudinal
    """
    
    def __init__(self, rbf_args: Dict[str, float] = dict(const=12, reg=1e-3)):
        """Constructor

        Args:
            points (ArrayLike): the (time x Npoints x 2) tracked points
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
            rbf_args (Dict[str, float], optional): Args for RBF instance.
                Defaults to dict(const=12, reg=1e-3).
        """
        ss = SessionState()
        
        self.deformation: np.ndarray = ss.deformation.value()
        
    def display(self):
        """Display results in streamlit application"""
        ss = SessionState()

        self.roi = ss.roi.value()

        # self.mesh.shape[1] != st.session_state.gl_strain.shape[2]
        if ss.strain.value() is None or st.sidebar.button('Recompute strain'):
            self.mesh, strain = EvaluationCase._strain(self.roi, self.deformation)
            ss.strain.update(strain)

        self.mesh = np.array(np.where(self.roi.T))
        self.strain = ss.strain.value()
        self.image = ss.image.value()[0]
        self.deformation = ss.deformation.value()

        st.write("""
            # Myocardial strain visualization
        """)

        with st.spinner('Preparing strain visualization...'):
            sa = StrainAnimation(self.image, self.deformation, self.mesh, self.strain)
            components.html(sa.anim.to_jshtml(), height=1000)


class StrainAnimation:

    def __init__(
        self, image: np.ndarray, deformation: np.ndarray, mesh: np.ndarray, strain: np.ndarray,
    ):

        self.image = image
        self.deformation = np.moveaxis(deformation, (0, 1, 2), (2, 0, 1))
        self.mesh = mesh
        self.strain = strain

        self.fig = plt.figure(figsize=(6, 6))
        gs = GridSpec(2, 2, figure=self.fig)
        self.ax_time = self.fig.add_subplot(gs[0, :])
        self.ax_circ = self.fig.add_subplot(gs[1, 0])
        self.ax_radi = self.fig.add_subplot(gs[1, 1])

        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        self.x, self.y = np.meshgrid(range(self.image.shape[1]), range(self.image.shape[0]))
        self.z = np.zeros((25, 2, *self.image.shape))

        for t in range(25):
            for i, (coords, s) in enumerate(zip(self.mesh.T, self.strain.swapaxes(0, 2))):
                m, n = tuple(coords)
                self.z[t, :, n, m] = s[[0, 1], [t, t]]

        self.mean_strain = self.strain.mean(axis=2)

        t = 0

        self.ax_time.plot(list(range(25)), self.mean_strain[:, 0], label='Circumferential', c='r')
        self.time_pt_c, = self.ax_time.plot(t, self.mean_strain[t, 0], marker='o', ms=8, c='r')

        self.ax_time.plot(list(range(25)), self.mean_strain[:, 1], label='Radial', c='b')
        self.time_pt_r, = self.ax_time.plot(t, self.mean_strain[t, 1], marker='o', ms=8, c='b')

        self.ax_time.legend(loc='best')

        self.ax_time.set_title('Avg. myocardial strain')
        self.ax_time.set_xlabel(r'Time-point $t$ (1)')
        self.ax_time.set_ylabel(r'Strain $E$ (%)')

        boundary = max(np.abs(self.strain[:, :2, :].min()), np.abs(self.strain[:, :2, :].max()))
        boundaries = dict(vmin=-boundary, vmax=boundary)

        self.mesh_circ = self.ax_circ.pcolormesh(
            self.x, self.y, np.ma.masked_where(self.z[t, 0] == 0, self.z[t, 0]), cmap='RdBu', **boundaries)
        self.ax_circ.set_title('Circumferential strain')

        self.mesh_radi = self.ax_radi.pcolormesh(
            self.x, self.y, np.ma.masked_where(self.z[t, 1] == 0, self.z[t, 1]), cmap='RdBu', **boundaries)
        self.ax_radi.set_title('Radial strain')

        cbar = plt.colorbar(self.mesh_radi, cax=self.fig.add_axes([0.13, 0.03, 0.02, 0.35]))
        cbar.set_label(r'Strain $E$ (%)')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        center_x, center_y = self.deformation.mean(axis=(0, 2))
        padding = 35
        for ax in [self.ax_circ, self.ax_radi]:
            ax.axis('off')
            ax.set_xlim(center_x - padding, center_x + padding)
            ax.set_ylim(center_y + padding, center_y - padding)

        plt.tight_layout()

        self.anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_animation,
            frames=25,
            interval=50,
            blit=True
        )

        plt.close()

    def init_animation(self):
        return self.animate(0)
        
    def animate(self, t):
        
        self.time_pt_c.set_data(t, self.mean_strain[t, 0])
        self.time_pt_r.set_data(t, self.mean_strain[t, 1])
        
        self.mesh_circ.set_array(np.ma.masked_where(self.z[t, 0] == 0, self.z[t, 0]))
        self.mesh_radi.set_array(np.ma.masked_where(self.z[t, 1] == 0, self.z[t, 1]))

        return [self.time_pt_r, self.time_pt_c, self.mesh_circ, self.mesh_radi]
