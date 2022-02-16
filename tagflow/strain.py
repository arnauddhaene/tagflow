import numpy as np

import streamlit as st

from .widgets.strain_estimator import StrainEstimator


def write():
    if st.session_state.points is not None and st.session_state.roi is not None:
        StrainEstimator(st.session_state.points.swapaxes(0, 2),
                        st.session_state.roi).display()
    else:
        st.warning('Please predict deformation field before estimating Green-Lagrangian strain.')
