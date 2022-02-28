from typing import Dict
from numpy.typing import ArrayLike

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px

import streamlit as st

from ..src.rbf import RBF, get_principle_strain
from .base import BaseWidget


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
    
    def __init__(self, points: ArrayLike, roi: ArrayLike,
                 rbf_args: Dict[str, float] = dict(const=12, reg=1e-3)):
        """Constructor

        Args:
            points (ArrayLike): the (Npoints x 2 x time) tracked points
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
            rbf_args (Dict[str, float], optional): Args for RBF instance.
                Defaults to dict(const=12, reg=1e-3).
        """
        
        self.cx, self.cy, self.radius = tuple(roi)
        
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

        x_idx, y_idx = tuple(map(lambda dim: np.arange(0, dim), st.session_state.image.shape[1:]))

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
        
        fig, ax = plt.subplots(2, 2)

        peak = np.argmax(np.abs(self.gl_strain.mean(axis=2))[:, 0])

        ax[0, 0].scatter(self.mesh[1], self.mesh[0], c=self.gl_strain[peak, 0, :], marker='o')
        ax[1, 0].scatter(self.mesh[1], self.mesh[0], c=self.gl_strain[peak, 1, :], marker='o')
        
        ax[0, 1].plot(range(self.Nt), self.gl_strain.mean(axis=2)[:, 0])
        ax[1, 1].plot(range(self.Nt), self.gl_strain.mean(axis=2)[:, 1])
        
        # data = pd.DataFrame({
        #     'Time': list(range(self.Nt)) + list(range(self.Nt)),
        #     'Strain': np.hstack([self.gl_strain.mean(axis=2)[:, 0],
        #                          self.gl_strain.mean(axis=2)[:, 1]]),
        #     'Direction': ['Circumferential'] * self.Nt + ['Radial'] * self.Nt
        # })
        
        # f = px.line(data, x='Time', y='Strain', facet_row='Direction')
        # f2 = px.scatter(x=self.mesh[1], y=self.mesh[0], color=self.gl_strain[peak, 0, :])
        
        # st.plotly_chart(f)
        # st.plotly_chart(f2) 
        
        st.pyplot(fig)
