from typing import Dict
from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

from ..rbf import RBF, get_principle_strain
from .base import BaseWidget


class StrainEstimator(BaseWidget):
    
    def __init__(self, points: ArrayLike, roi: ArrayLike,
                 rbf_args: Dict[str, float] = dict(const=12, reg=1e-3)):
        
        self.cx, self.cy, self.radius = tuple(roi)
        # Define roi_centre with call to `mask`
        self.mesh = self.compute_mesh()
        
        self.points = np.swapaxes(points - self.roi_centre[:, :, None], 0, 1)
        
        self.Nt = self.points.shape[2]
        self.Np = self.mesh.shape[1]
        
        self.rbf = RBF(self.points[:, :, 0], **rbf_args)
                
    def compute_mesh(self):

        x_idx, y_idx = tuple(map(lambda dim: np.arange(0, dim), st.session_state.image.shape[1:]))

        mask = (x_idx[:, np.newaxis] - self.cx) ** 2 + (y_idx[np.newaxis, :] - self.cy) ** 2 \
            < (.95 * self.radius) ** 2
        lv_mask = (x_idx[:, np.newaxis] - self.cx) ** 2 + (y_idx[np.newaxis, :] - self.cy) ** 2 \
            < (.4 * self.radius) ** 2

        myocardial_mask = mask ^ lv_mask

        mask_idx = np.array(np.where(myocardial_mask)).transpose()
        
        self.roi_centre = mask_idx.mean(axis=0)[:, None].transpose()
        
        return (mask_idx - self.roi_centre).T
    
    def solve(self):
        
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
                
        if 'gl_strain' not in st.session_state:
            st.session_state.gl_strain = self.solve()
        self.gl_strain = st.session_state.gl_strain
        
        fig, ax = plt.subplots(2, 2)

        peak = np.argmax(np.abs(self.gl_strain.mean(axis=2))[:, 0])

        ax[0, 0].scatter(self.mesh[1], self.mesh[0], c=self.gl_strain[peak, 0, :], marker='o')
        ax[1, 0].scatter(self.mesh[1], self.mesh[0], c=self.gl_strain[peak, 1, :], marker='o')
        
        ax[0, 1].plot(range(self.Nt), self.gl_strain.mean(axis=2)[:, 0])
        ax[1, 1].plot(range(self.Nt), self.gl_strain.mean(axis=2)[:, 1])
        
        st.pyplot(fig)
