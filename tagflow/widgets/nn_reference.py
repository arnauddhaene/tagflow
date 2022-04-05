import numpy as np
from skimage import morphology, measure

import torch

from .reference_picker import ReferencePicker
from ..models.segmentation.unet_utils import _preprocess_image, _postprocess_mask
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

        self.preprocessor = _preprocess_image()

    def preprocess(self) -> torch.Tensor:
        """Preprocessing pipeline before model inference

        Returns:
            torch.Tensor: Model input in (B, C, W, H)
        """
        return self.preprocessor(self.image[0].astype(np.float64))
    
    def reference(self):
        """Computes reference tracking points
        
        Modifies:
            ref_points (ArrayLike): reference tracking points (Npoints x 2)
            roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
        """
        
        inp = self.preprocess().unsqueeze(0)
                
        prediction = segment(inp)

        out = _postprocess_mask(self.image[0].shape)(prediction[0])[0].numpy()
        out = (out == 1)  # Select MYO class
        out = morphology.binary_closing(out)  # Close segmentation mask
        blobs, num = measure.label(out, background=0, return_num=True)  # Closed components
        sizes = [(blobs == i).sum() for i in range(1, num + 1)]  # Evaluate component size
        blob_index = np.argmax(sizes) + 1  # Fetch index of largest blob
        
        self.roi = (blobs == blob_index)
               
        self.ref_points = EvaluationCase._reference(np.array(self.roi))
