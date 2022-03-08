import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type:ignore

from .reference_picker import ReferencePicker


class ManualReference(ReferencePicker):
    """The manual annotation reference tracking point picker

    Attributes:
        image (ArrayLike): the (T x W x H) input image
        ref_points (ArrayLike): the (Npoints x 2) reference tracking points
        roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
        stretch (int): stretch factor for displaying the drawable canvas
        drawed_annot (Dict[Any, Any]): save-able annotation from drawable canvas
    """
    
    def __init__(self, image: np.ndarray, stretch: int = 6):
        """Constructor

        Args:
            image (np.ndarray): the (T x W x H) input image
            stretch (int, optional): stretch factor for displaying the drawable canvas.
                Defaults to 6.
        """
        super().__init__(image)
        
        self.stretch = stretch
        
        if 'drawed_annot' not in st.session_state:
            st.session_state.drawed_annot = None
        self.drawed_annot = st.session_state.drawed_annot
                
    def preprocess(self) -> Image:
        """Preprocessing pipeline to edit canvas background for better annotation
        
        Modifies:
            xmin, xmax, ymin, ymax (int): coordinates of zoomed in image view

        Returns:
            PIL.Image: Image to use as background for drawable canvas
        """
        
        xdim, ydim = tuple(self.image.shape[1:])
        
        self.xmin, self.xmax = tuple(map(int, (xdim * .2, xdim * .7,)))
        self.ymin, self.ymax = tuple(map(int, (ydim * .2, ydim * .7,)))
        
        contrast = st.sidebar.slider('Contrast', .5, 5., 1.25)
        brightness = st.sidebar.slider('Brightness', .5, 5., 1.25)
        
        processed = Image.fromarray(self.image[0, self.xmin:self.xmax, self.ymin:self.ymax])
        processed = processed.convert(mode='RGB')

        processed = ImageEnhance.Contrast(processed).enhance(contrast)
        processed = ImageEnhance.Brightness(processed).enhance(brightness)
        
        return processed
    
    def reference(self):
        """Computes reference tracking points
        
        Modifies:
            ref_points (ArrayLike): reference tracking points (Npoints x 2)
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
            drawed_annot (Dict[Any, Any]): save-able annotation from drawable canvas
        """
        
        canvas = st_canvas(
            fill_color='#FF0000', stroke_color='#FF0000',
            stroke_width=1., point_display_radius=3., drawing_mode='point',
            background_image=self.preprocess(), update_streamlit=True,
            height=self.stretch * (self.xmax - self.xmin),
            width=self.stretch * (self.ymax - self.ymin),
            initial_drawing=self.drawed_annot
        )
        
        if canvas.json_data is not None:
            self.drawed_annot = canvas.json_data
            objects = pd.json_normalize(self.drawed_annot['objects'])
            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype('str')
                
            if len(objects) > 0:
                lt = np.array(objects[['left', 'top']])
                r = np.array(objects['radius'])  # radius
                sw = np.array(objects['strokeWidth']) / 2.  # half of strokeWidth
                
                offset = np.vstack([r + sw, np.zeros_like(r)]).T
                                
                self.ref_points = np.array([self.ymin, self.xmin]) \
                    + (lt + offset) / self.stretch
                
                centre = self.ref_points.mean(axis=0).T
                radius = 1.1 * np.abs(np.linalg.norm(centre - self.ref_points, axis=1)).max()
                
                self.roi = np.array([*centre, radius])
                
    def plot(self):
        """No need to plot the image here as it serves as the background of the
        drawable canvas
        """
        pass
            
    def save_reference(self):
        """Extends saving to session state with drawed_annot
        """
        super().save_reference()
        st.session_state.drawed_annot = self.drawed_annot
        
    def clear_reference(self):
        """Extends clearing session state with drawed_annot
        """
        super().clear_reference()
        st.session_state.drawed_annot = None
