from typing import Dict

import string

import numpy as np
import matplotlib.pyplot as plt

from skimage import measure

import streamlit as st

from .base import BaseWidget
from ..state.state import SessionState
from ..src.case import EvaluationCase
from tagflow.widgets.reference_picker import ReferencePicker


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

        self.mesh = np.array(np.where(self.roi))
        self.strain = ss.strain.value()
        self.image = ss.image.value()[0]

        x, y = np.meshgrid(range(self.image.shape[1]), range(self.image.shape[0]))
        z = np.zeros((2, *self.image.shape))

        peaks = np.abs(self.strain.swapaxes(0, 2).mean(axis=0))[:2].argmax(axis=1)

        for i, (coords, s) in enumerate(zip(self.mesh.T, self.strain.swapaxes(0, 2))):
            m, n = tuple(coords)
            z[:, m, n] = s[[0, 1], peaks]

        deformation = np.moveaxis(ss.deformation.value(), (0, 1, 2), (2, 0, 1))

        fig, ax = plt.subplots(2, 3, figsize=(10, 6.5))

        ax[0, 0].imshow(self.image, cmap='gray')
        # ax[0, 0].imshow(np.ma.masked_where(self.roi == 0, self.roi), cmap='Reds', alpha=0.4)
        contours = measure.find_contours(self.roi)
        for i, contour in enumerate(contours):
            contour = ReferencePicker.interp_pts(contour[::5, ::-1])
            ax[0, 0].plot(*contour, c='#C44536', label='Myocardium' if i == 0 else None)

        ax[0, 0].set_title('Myocardium segmentation')

        ax[0, 1].imshow(self.image, cmap='gray')
        for i in range(deformation.shape[0]):
            ax[0, 1].plot(deformation[i, 0, :], deformation[i, 1, :], color='y')
        ax[0, 1].set_title('Deformation field')

        mean_strain = self.strain.mean(axis=2)
        ax[0, 2].plot(list(range(25)), mean_strain[:, 0], label='Circumferential', c='r')
        ax[0, 2].plot(peaks[0], mean_strain[peaks[0], 0], marker='o', ms=8, c='r')
        ax[0, 2].axvline(peaks[0], linestyle=':', c='r')
        ax[0, 2].plot(list(range(25)), mean_strain[:, 1], label='Radial', c='b')
        ax[0, 2].plot(peaks[1], mean_strain[peaks[1], 1], marker='o', ms=8, c='b')
        ax[0, 2].axvline(peaks[1], linestyle=':', c='b')
        ax[0, 2].legend(loc='best')
        ax[0, 2].set_title('Avg. myocardial strain')
        ax[0, 2].set_xlabel(r'Time-point $t$ (1)')
        ax[0, 2].set_ylabel(r'Strain $E$ (%)')

        boundaries = dict(vmin=self.strain[:, :2, :].min(), vmax=self.strain[:, :2, :].max())
        mesh1 = ax[1, 1].pcolormesh(x, y, np.ma.masked_where(z[0] == 0, z[0]), cmap='RdBu', **boundaries)
        ax[1, 1].set_title(r'Circumferential strain $t_{%s}$' % peaks[0])

        mesh2 = ax[1, 2].pcolormesh(x, y, np.ma.masked_where(z[1] == 0, z[1]), cmap='RdBu', **boundaries)
        ax[1, 2].set_title(r'Radial strain $t_{%s}$' % peaks[1])

        center_x, center_y = deformation.mean(axis=(0, 2))
        padding = 35
        off_axes = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 0)]
        for a, b in off_axes:
            ax[a, b].axis('off')
            ax[a, b].set_xlim(center_x - padding, center_x + padding)
            ax[a, b].set_ylim(center_y + padding, center_y - padding)

        for n, axes in enumerate(np.array(ax.flat)[[0, 1, 2, 4, 5]]):
            axes.text(-0.15, 1.13, string.ascii_uppercase[n], 
                      transform=axes.transAxes, size=20, weight='bold')

        plt.colorbar(mesh1, ax=ax[1, 1])
        plt.colorbar(mesh2, ax=ax[1, 2])

        plt.tight_layout()
        st.pyplot(fig)
