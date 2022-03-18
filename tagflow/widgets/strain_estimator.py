from typing import Dict

import numpy as np
import holoviews as hv

import streamlit as st

from .base import BaseWidget
from ..state.state import SessionState
from ..src.case import EvaluationCase


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
        self.deformation: np.ndarray = ss.deformation.value()
        
    def display(self):
        """Display results in streamlit application"""
        ss = SessionState()

        # self.mesh.shape[1] != st.session_state.gl_strain.shape[2]
        if ss.strain.value() is None:
            self.mesh, strain = EvaluationCase._strain(ss.roi.value(), self.deformation)
            ss.strain.update(strain)

        self.mesh = np.array(np.where(ss.roi.value()))
        self.gl_strain = ss.strain.value()
        
        peaks = np.argmax(np.abs(self.gl_strain.mean(axis=2)), axis=0)
                
        times = list(range(self.gl_strain.shape[0]))
        
        time = hv.Dimension('time', label='Time', unit='s')
        strain_c = hv.Dimension('strain_c', label='Circumferential strain')
        strain_r = hv.Dimension('strain_c', label='Radial strain')

        cir = hv.Points((self.mesh[1], -self.mesh[0], self.gl_strain[peaks[0], 0, :]),
                        vdims='strain', group='Peak circumferential strain', label=f'(t={peaks[0]}')
        rad = hv.Points((self.mesh[1], -self.mesh[0], self.gl_strain[peaks[1], 1, :]),
                        vdims='strain', group='Peak radial strain', label=f'(t={peaks[1]}')

        cir_t = hv.Curve((times, self.gl_strain.mean(axis=2)[:, 0]), time, strain_c, label='Circumferential')
        rad_t = hv.Curve((times, self.gl_strain.mean(axis=2)[:, 1]), time, strain_r, label='Radial')
            
        fig = (cir + rad).opts(hv.opts.Points(color='strain', cmap='viridis', marker='square',
                                              size=4, colorbar=True, xaxis=None, yaxis=None)) \
            + (cir_t * rad_t).opts(ylabel=r"$$E$$", legend_position='top')

        st.bokeh_chart(hv.render(fig.cols(3), backend='bokeh'))
