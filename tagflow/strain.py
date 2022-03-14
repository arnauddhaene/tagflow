import streamlit as st

from .widgets.strain_estimator import StrainEstimator
from .state.state import SessionState, SessionStatus


def write():
    
    ss = SessionState()
        
    if ss.status().value < SessionStatus.deformation.value:
        st.warning('Please predict deformation field before estimating Green-Lagrangian strain.')
    else:
        StrainEstimator().display()
