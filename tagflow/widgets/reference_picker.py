from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type:ignore
from scipy import interpolate  # , spatial
from skimage import draw


from .base import BaseWidget
from ..src.case import EvaluationCase
from ..state.state import SessionState
from ..utils import unpack_circle


class ReferencePicker(BaseWidget):
    """Parent Widget class for a reference tracking point set picker

    Attributes:
        image (np.ndarray): the (T x W x H) input image
        ref_points (np.ndarray): the (Npoints x 2) reference tracking points
        roi (np.ndarray): circle coordinates for outer ROI [Cx, Cy, R]
    """
    
    def __init__(self, stretch: float = 6, aspect: float = .6):
        """Constructor"""
        ss = SessionState()
        
        self.canvas = None
        self.stretch = stretch
        self.aspect = aspect
        
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

    def compute_ref_points(self):
        return EvaluationCase._reference(
            np.array(self.roi),
            self.reference_method,
            self.image[0].astype(np.float32)
        )
        
    def display(self):
        """Display in streamlit application"""

        canvas_args = {}

        mode = st.sidebar.selectbox('Drawing mode', ['transform', 'point'])
        self.reference_method = st.sidebar.selectbox(
            'Reference setting method', ['intersections', 'mesh'],
        )

        canvas_args['drawing_mode'] = mode
        if mode == 'point':
            contour = st.sidebar.selectbox('Contour', ['inner', 'outer'])
            canvas_args['stroke_color'] = '#FF0000' if contour == 'inner' else '#0000FF'
            canvas_args['fill_color'] = '#FF0000' if contour == 'inner' else '#0000FF'

        save, clear = st.sidebar.columns(2)

        save.button('Save reference', on_click=self.save_reference)
            
        if SessionState().reference.value() is not None:
            clear.button('Clear reference', on_click=self.clear_reference)
        
        bg_image: Image = self.enhance()

        if (self.ref_points is None or self.contour is None) and self.roi is None:
            self.reference()
        
        self.canvas = st_canvas(
            **canvas_args,
            stroke_width=1., point_display_radius=3.,
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
        shortest_dim = min(xdim, ydim)
        
        cx, cy = xdim / 2, ydim / 2

        self.xmin, self.xmax = int(cx - (shortest_dim / 5)), int(cx + (shortest_dim / 5))
        self.ymin, self.ymax = \
            int(cy - (shortest_dim / (5 * self.aspect))), int(cy + (shortest_dim / (5 * self.aspect)))
        
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
        
        if self.contour is not None:

            outer, inner = self.contour

            # use offset and ymin, xmin to push coords into canvas space
            offset = np.array([3.5, 0.0])
            ref_inner = (inner - np.array([self.ymin, self.xmin])) * self.stretch - offset
            ref_outer = (outer - np.array([self.ymin, self.xmin])) * self.stretch - offset
            
            return dict(
                objects=([
                    dict(
                        type='circle', originX='left', originY='center',
                        left=left, top=top, width=6, height=6, fill='#FF0000', stroke='#FF0000',
                        strokeWidth=1, angle=0, paintFirst='fill', radius=3
                    ) for left, top in ref_inner
                ] + [
                    dict(
                        type='circle', originX='left', originY='center',
                        left=left, top=top, width=6, height=6, fill='#0000FF', stroke='#0000FF',
                        strokeWidth=1, angle=0, paintFirst='fill', radius=3
                    ) for left, top in ref_outer
                ])
            )

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
        self.contour = ss.contour.value()

    def save_reference(self):
        """Save reference tracking points to sessions state"""
        if self.canvas is not None and self.canvas.json_data is not None:
            self.drawed_annot = self.canvas.json_data
            objects = pd.json_normalize(self.drawed_annot['objects'])
            # convert object columns to str for reading and processing
            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype('str')
                
            if len(objects) > 0:

                self.contour = []

                for color in ['#0000FF', '#FF0000']:

                    lt = np.array(objects[objects.stroke == color][['left', 'top']])
                    r = np.array(objects[objects.stroke == color]['radius'])  # radius
                    sw = np.array(objects[objects.stroke == color]['strokeWidth']) / 2.  # half of strokeWidth
                    
                    offset = np.vstack([r + sw, np.zeros_like(r)]).T
                                                    
                    self.contour.append(
                        np.array([self.ymin, self.xmin]) + (lt + offset) / self.stretch
                    )
                    
                shape = self.image.shape[1:]
                self.roi = ReferencePicker.compute_roi(self.contour, shape)
                self.ref_points = self.compute_ref_points()

        ss = SessionState()
        ss.reference.update(self.ref_points)
        ss.roi.update(self.roi)
        ss.contour.update(self.contour)
        
    def clear_reference(self):
        """Clear reference tracking points and def and roi from sessions state"""
        ss = SessionState()
        ss.clear(['reference', 'deformation', 'roi'])

    def refresh_ref_points(self):
        """Clear only reference tracking points from sessions state"""
        ss = SessionState()
        ss.clear(['reference'])
        ss.reference.update(self.compute_ref_points())

    @staticmethod
    def compute_roi(contour: List[np.ndarray], size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        
        # # Outer hull
        # outer_hull = spatial.ConvexHull(contour)
        
        # # Inner hull by finding the Convex Hull of inverted radius space
        # t0 = contour - contour.mean(axis=0)
        # x, y = t0[:, 0], t0[:, 1]
        # t0 = np.array([np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)]).T
        # t0[:, 0] = 1 / (t0[:, 0] + 1e-12)
        # r, t = t0[:, 0], t0[:, 1]
        # t0 = np.array([r * np.sin(t), r * np.cos(t)]).T

        # inner_hull = spatial.ConvexHull(t0)
        
        outer_pts = ReferencePicker.interp_pts(contour[0])
        inner_pts = ReferencePicker.interp_pts(contour[1])
                
        inner_pg = draw.polygon(inner_pts[1, :], inner_pts[0, :], size)
        outer_pg = draw.polygon(outer_pts[1, :], outer_pts[0, :], size)
        
        inner_mask = draw.polygon2mask(size, np.array(inner_pg).T)
        outer_mask = draw.polygon2mask(size, np.array(outer_pg).T)
        
        return outer_mask ^ inner_mask
        
    @staticmethod
    def interp_pts(points: np.ndarray) -> np.ndarray:
        # From Mike Loecher
        x, y = points[:, 0], points[:, 1]
        x, y = np.hstack((x, x[0])), np.hstack((y, y[0]))

        dxy = np.hypot(np.diff(x), np.diff(y))

        tt = np.hstack((0, np.cumsum(dxy)))
        tt /= tt.max()

        spl_x = interpolate.splrep(tt, x, k=5, s=0, per=True)
        spl_y = interpolate.splrep(tt, y, k=5, s=0, per=True)

        tt_new = np.linspace(0, 1, 100)

        x_out = interpolate.splev(tt_new, spl_x)
        y_out = interpolate.splev(tt_new, spl_y)
        
        return np.array([x_out, y_out])
