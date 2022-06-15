import numpy as np
from skimage import measure

from .reference_picker import ReferencePicker
from ..src.predict import segment
from ..src.case import EvaluationCase


class NeuralReference(ReferencePicker):
    """The tagged MRI ROI Segmentation and reference point picker.

    Attributes:
        image (ArrayLike): the (T x W x H) input image
        ref_points (ArrayLike): the (Npoints x 2) reference tracking points
        roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
    """
    
    def __init__(self):
        """Constructor"""
        super().__init__()
    
    def reference(self):
        """Computes reference tracking points
        
        Modifies:
            ref_points (ArrayLike): reference tracking points (Npoints x 2)
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
        """
        self.roi = segment(self.image[0])
        contour = list(map(lambda c: c[::15, ::-1], measure.find_contours(self.roi)))
        # self.contour = np.concatenate(contour)
        self.contour = contour
        self.ref_points = EvaluationCase._reference(np.array(self.roi))
        
        self.save_reference()
