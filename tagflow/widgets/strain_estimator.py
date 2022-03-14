from typing import Dict
from numpy.typing import ArrayLike

import numpy as np
import holoviews as hv

import streamlit as st

from ..src.rbf import RBF, get_principle_strain
from .base import BaseWidget
from ..state.state import SessionState


hv.extension('bokeh')


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
            image (ArrayLike): image
            rbf_args (Dict[str, float], optional): Args for RBF instance.
                Defaults to dict(const=12, reg=1e-3).
        """
        ss = SessionState()
        
        self.image: np.ndarray = ss.image.value()
        
        points = ss.deformation.value()
        self.cx, self.cy = points.mean(axis=(0, 1))
        
        self.mesh: np.ndarray = self.compute_mesh(ss.roi.value())
        # Needs to be (2 x Npoints time) to be solved more efficiently
        self.deformation: np.ndarray = np.swapaxes(points - np.array([self.cx, self.cy])[None, None, :], 0, 2)
                                                           
        self.Nt: int = self.deformation.shape[2]
        self.Np: int = self.mesh.shape[1]

        self.rbf: RBF = RBF(self.deformation[:, :, 0], **rbf_args)
                
    def compute_mesh(self, roi: np.ndarray) -> np.ndarray:
        """Compute set of pixel coordinates in mesh from LV ROI and pseudo-wall estimation

        Args:
            roi (np.ndarray): Binary mask of LV myocardium (W x H).

        Returns:
            ArrayLike: Set of all pixel coordinates within LV wall mask (2 x Npoints) relative
                to ROI centre
        """
        mask_idx = np.array(np.where(roi)).T
        
        return (mask_idx - np.array([self.cx, self.cy])).T
    
    def solve(self) -> ArrayLike:
        """Solve Radial Bias Function interpolation

        Returns:
            ArrayLike: the computed strain (time x dir x point) with dirs circumferential,
            radial, and longitudinal
        """
        
        gl_strain = []

        for time in range(self.Nt):
            points_t = self.deformation[:, :, time]
            deformation_grad = np.zeros([self.Np, 3, 3])
            
            for dim in range(2):
                _ = self.rbf.solve(points_t[dim, :])
                deformation_grad[:, dim, :2] = self.rbf.derivative(self.mesh).T
                
            gl_strain.append(get_principle_strain(self.mesh, deformation_grad))
            
        return np.array(gl_strain)
        
    def display(self):
        """Display results in streamlit application"""
        ss = SessionState()

        # self.mesh.shape[1] != st.session_state.gl_strain.shape[2]
        # if ss.strain.value() is None:
        ss.strain.update(self.solve())

        self.gl_strain = ss.strain.value()
        
        peaks = np.argmax(np.abs(self.gl_strain.mean(axis=2)), axis=0)
                
        times = list(range(self.Nt))

        time = hv.Dimension('time', label='Time', unit='s')
        strain_c = hv.Dimension('strain_c', label='Circumferential strain')
        strain_r = hv.Dimension('strain_c', label='Radial strain')

        cir = hv.Points((self.mesh[1], -self.mesh[0], self.gl_strain[peaks[0], 0, :]),
                        vdims='strain') \
            .options(color='strain', cmap='viridis', marker='square', size=4,
                     title=f'Peak circumferential strain (time={peaks[0]})',
                     xaxis=None, yaxis=None, colorbar=True)
        rad = hv.Points((self.mesh[1], -self.mesh[0], self.gl_strain[peaks[1], 1, :]),
                        vdims='strain') \
            .options(color='strain', cmap='viridis', marker='square', size=4,
                     title=f'Peak radial strain (time={peaks[1]})',
                     xaxis=None, yaxis=None, colorbar=True)

        cir_t = hv.Curve((times, self.gl_strain.mean(axis=2)[:, 0]), time, strain_c) \
            .opts(ylabel=r"$$E_{cc}$$", title='Circumferential strain in time')
        rad_t = hv.Curve((times, self.gl_strain.mean(axis=2)[:, 1]), time, strain_r) \
            .opts(ylabel=r"$$E_{rr}$$", title='Radial strain in time')

        st.bokeh_chart(hv.render((cir + cir_t + rad + rad_t).cols(2), backend='bokeh'))
