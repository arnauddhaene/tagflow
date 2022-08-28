import streamlit as st

from .widgets.hough_reference import HoughReference
from .widgets.nn_reference import NeuralReference
from .state.state import SessionState


def write():
    ss = SessionState()

    if ss.image.value() is not None:
        
        annot = st.sidebar.selectbox('Annotation', ['Neural Network', 'Hough Transform'])
        
        st.write("""
            #  Reference Setting

            Select the annotation method in the sidebar. Points will automatically be placed
            around inner and outer contours. Use the **transform** and **point** drawing
            modes to  drag and drop existing points or add additional ones respectively.
        """)
        col1, col2 = st.columns(2)
        stretch = col1.number_input('Image size', 2., 7., 6., .5)
        aspect = col2.number_input('Padding from ref center', .5, 1.5, .6, .1)
        
        if annot == 'Hough Transform':
            HoughReference(stretch, aspect).display()
        elif annot == 'Neural Network':
            NeuralReference().display()
        
    else:
        st.warning('Please upload an image first.')
