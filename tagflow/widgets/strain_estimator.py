from typing import Dict
from numpy.typing import ArrayLike

import numpy as np
import holoviews as hv

import streamlit as st

from ..src.rbf import RBF, get_principle_strain
from .base import BaseWidget


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
    
    def __init__(self, points: ArrayLike, roi: ArrayLike, image: ArrayLike,
                 rbf_args: Dict[str, float] = dict(const=12, reg=1e-3)):
        """Constructor

        Args:
            points (ArrayLike): the (Npoints x 2 x time) tracked points
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
            image (ArrayLike): image
            rbf_args (Dict[str, float], optional): Args for RBF instance.
                Defaults to dict(const=12, reg=1e-3).
        """
        
        self.cx, self.cy, self.radius = tuple(roi)
        self.image = image
        
        self.mesh = self.compute_mesh()
        
        self.points = np.swapaxes(points - np.array([self.cx, self.cy])[None, :, None], 0, 1)
        
        self.Nt = self.points.shape[2]
        self.Np = self.mesh.shape[1]
        
        self.rbf = RBF(self.points[:, :, 0], **rbf_args)
                
    def compute_mesh(self) -> ArrayLike:
        """Compute set of pixel coordinates in mesh from LV ROI and pseudo-wall estimation

        Returns:
            ArrayLike: Set of all pixel coordinates within LV wall mask (2 x Npoints) relative
                to ROI centre
        """

        x_idx, y_idx = tuple(map(lambda dim: np.arange(0, dim), self.image.shape[1:]))

        mask = (x_idx[:, np.newaxis] - self.cx) ** 2 + (y_idx[np.newaxis, :] - self.cy) ** 2 \
            < (.95 * self.radius) ** 2
        wall_mask = (x_idx[:, np.newaxis] - self.cx) ** 2 + (y_idx[np.newaxis, :] - self.cy) ** 2 \
            < (.4 * self.radius) ** 2

        myocardial_mask = mask ^ wall_mask

        mask_idx = np.array(np.where(myocardial_mask)).transpose()
        
        return (mask_idx - np.array([self.cx, self.cy])).T
    
    def solve(self) -> ArrayLike:
        """Solve Radial Bias Function interpolation

        Returns:
            ArrayLike: the computed strain (time x dir x point) with dirs circumferential,
            radial, and longitudinal
        """
        
        gl_strain = []

        for time in range(self.Nt):
            points_t = self.points[:, :, time]
            deformation_grad = np.zeros([self.Np, 3, 3])
            
            for dim in range(2):
                _ = self.rbf.solve(points_t[dim, :])
                deformation_grad[:, dim, :2] = self.rbf.derivative(self.mesh).T
                
            gl_strain.append(get_principle_strain(self.mesh, deformation_grad))
            
        return np.array(gl_strain)
        
    def display(self):
        """Display results in streamlit application"""

        if ('gl_strain' not in st.session_state) or \
                (self.mesh.shape[1] != st.session_state.gl_strain.shape[2]):
            st.session_state.gl_strain = self.solve()
        self.gl_strain = st.session_state.gl_strain
        
        peaks = np.argmax(np.abs(self.gl_strain.mean(axis=2)), axis=0)
                
        times = list(range(self.Nt))

        time = hv.Dimension('time', label='Time', unit='s')
        strain_c = hv.Dimension('strain_c', label='Circumferential strain')
        strain_r = hv.Dimension('strain_c', label='Radial strain')

        cir = hv.Points((self.mesh[1], self.mesh[0], self.gl_strain[peaks[0], 0, :]),
                        vdims='strain') \
            .options(color='strain', cmap='viridis', marker='square', size=4,
                     title=f'Peak circumferential strain (time={peaks[0]})',
                     xaxis=None, yaxis=None, colorbar=True)
        rad = hv.Points((self.mesh[1], self.mesh[0], self.gl_strain[peaks[1], 1, :]),
                        vdims='strain') \
            .options(color='strain', cmap='viridis', marker='square', size=4,
                     title=f'Peak radial strain (time={peaks[1]})',
                     xaxis=None, yaxis=None, colorbar=True)

        cir_t = hv.Curve((times, self.gl_strain.mean(axis=2)[:, 0]), time, strain_c) \
            .opts(ylabel=r"$$E_{cc}$$", title='Circumferential strain in time')
        rad_t = hv.Curve((times, self.gl_strain.mean(axis=2)[:, 1]), time, strain_r) \
            .opts(ylabel=r"$$E_{rr}$$", title='Radial strain in time')

        st.bokeh_chart(hv.render((cir + cir_t + rad + rad_t).cols(2), backend='bokeh'))
