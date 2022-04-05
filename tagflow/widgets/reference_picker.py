from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type:ignore

from .base import BaseWidget
from ..state.state import SessionState
from ..utils import unpack_circle


class ReferencePicker(BaseWidget):
    """Parent Widget class for a reference tracking point set picker

    Attributes:
        image (np.ndarray): the (T x W x H) input image
        ref_points (np.ndarray): the (Npoints x 2) reference tracking points
        roi (np.ndarray): circle coordinates for outer ROI [Cx, Cy, R]
    """
    
    def __init__(self):
        """Constructor"""
        ss = SessionState()
        
        self.canvas = None
        self.stretch = 6
        
        self.image: np.ndarray = ss.image.value()
        self.ref_points: Optional[np.ndarray] = None
        self.roi: Optional[np.ndarray] = None
        
        self.load_reference()
    
    def reference(self):
        """This method should compute self.ref_points and self.roi
        
        Modifies:
            ref_points (ArrayLike): reference tracking points (Npoints x 2)
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]

        Raises:
            NotImplementedError: forces children classes to implement this
        """
        raise NotImplementedError()
        
    def display(self):
        """Display in streamlit application"""

        mode = st.sidebar.selectbox('Drawing mode', ['transform', 'point'])
        save, clear = st.sidebar.columns(2)

        save.button('Save reference', on_click=self.save_reference)
            
        if SessionState().reference.value() is not None:
            clear.button('Clear reference', on_click=self.clear_reference)
        
        bg_image: Image = self.enhance()

        if self.ref_points is None and self.roi is None:
            self.reference()
            self.save_reference()
        
        self.canvas = st_canvas(
            fill_color='#FF0000', stroke_color='#FF0000',
            stroke_width=1., point_display_radius=3., drawing_mode=mode,
            background_image=bg_image, update_streamlit=True,
            height=self.stretch * (self.xmax - self.xmin),
            width=self.stretch * (self.ymax - self.ymin),
            initial_drawing=self.fetch_drawed_annot()
        )
        
    def enhance(self) -> Image:
        """Preprocessing pipeline to edit canvas background for better annotation
        
        Modifies:
            xmin, xmax, ymin, ymax (int): coordinates of zoomed in image view

        Returns:
            PIL.Image: Image to use as background for drawable canvas
        """
        
        xdim, ydim = tuple(self.image.shape[1:])
        
        self.xmin, self.xmax = tuple(map(int, (xdim * .30, xdim * .65,)))
        self.ymin, self.ymax = tuple(map(int, (ydim * .25, ydim * .75,)))
        
        ct, bn = st.sidebar.columns(2)
        
        contrast = ct.slider('Contrast', .5, 5., 1.25)
        brightness = bn.slider('Brightness', .5, 5., 1.25)
        
        processed = Image.fromarray(self.image[0, self.xmin:self.xmax, self.ymin:self.ymax])
        processed = processed.convert(mode='RGB')

        processed = ImageEnhance.Contrast(processed).enhance(contrast)
        processed = ImageEnhance.Brightness(processed).enhance(brightness)
        
        # Add mask of roi
        if self.roi is not None:
            mask = (self.roi[self.xmin:self.xmax, self.ymin:self.ymax] * 85)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask, mode='L')
            
            blue = Image.new('RGB', mask.size, (128, 0, 128))
            
            processed = Image.composite(blue, processed, mask)
        
        return processed
    
    def fetch_drawed_annot(self) -> Dict[str, Any]:
        
        if self.ref_points is not None:
            # use offset and ymin, xmin to push coords into canvas space
            offset = np.array([3.5, 0.0])
            ref = (self.ref_points - np.array([self.ymin, self.xmin])) * self.stretch - offset
            
            return dict(objects=[
                dict(
                    type='circle', originX='left', originY='center',
                    left=left, top=top, width=6, height=6, fill='#FF0000', stroke='#FF0000',
                    strokeWidth=1, angle=0, paintFirst='fill', radius=3
                ) for left, top in ref
            ])

    def zoom_in(self):
        """Zooms in on the plotted image based on center or padded reference tracking points."""
        
        height, width = tuple(self.image.shape[1:])

        if self.ref_points is None:
            xmin, xmax = width * .25, width * .75
            ymin, ymax = height * .25, height * .75
        else:
            xmin, xmax = self.ref_points[:, 0].min(), self.ref_points[:, 0].max()
            ymin, ymax = self.ref_points[:, 1].min(), self.ref_points[:, 1].max()
        
        plt.xlim(max(0, xmin - 20), min(xmax + 20, width))
        plt.ylim(min(ymax + 20, height), max(0, ymin - 20))
        
    @staticmethod
    def circle_mask(circle: np.ndarray, shape: Tuple[int, int], rho: float = 1.) -> np.ndarray:
        cx, cy, radius = unpack_circle(circle)
        x_idx, y_idx = [np.arange(0, dim) for dim in iter(shape)]
        mask = (x_idx[:, np.newaxis] - cx) ** 2 + (y_idx[np.newaxis, :] - cy) ** 2 < (rho * radius) ** 2
        return mask.T

    def load_reference(self):
        """Load reference tracking points from sessions state"""
        ss = SessionState()
        self.ref_points = ss.reference.value()
        self.roi = ss.roi.value()

    def save_reference(self):
        """Save reference tracking points to sessions state"""
        if self.canvas is not None and self.canvas.json_data is not None:
            self.drawed_annot = self.canvas.json_data
            objects = pd.json_normalize(self.drawed_annot['objects'])
            # convert object columns to str for reading and processing
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
                
                circle = np.array([*centre, radius])
                shape = self.image.shape[1:]
                self.roi = self.circle_mask(circle, shape, 0.9) ^ \
                    self.circle_mask(circle, shape, 0.5)
        
        ss = SessionState()
        ss.reference.update(self.ref_points)
        ss.roi.update(self.roi)
        
    def clear_reference(self):
        """Clear reference tracking points from sessions state"""
        ss = SessionState()
        ss.clear(['reference', 'deformation', 'roi'])
