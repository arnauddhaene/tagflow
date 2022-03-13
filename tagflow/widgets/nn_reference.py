import numpy as np

import torch

from .reference_picker import ReferencePicker
from ..models.segmentation.unet_utils import _preprocess_image, _postprocess_mask
from ..src.predict import segment


class NeuralReference(ReferencePicker):
    """The tagged MRI ROI Segmentation and reference point picker.

    Attributes:
        image (ArrayLike): the (T x W x H) input image
        ref_points (ArrayLike): the (Npoints x 2) reference tracking points
        roi (ArrayLike): circle coordinates for outer ROI [Cx, Cy, R]
    """
    
    def __init__(self, image: np.ndarray):
        """Constructor d

        Args:
            image (ArrayLike): the (T x W x H) input image
            kernel (int, optional): Blurring kernel size. Defaults to 5.
        """
        super().__init__(image)

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

        self.roi = _postprocess_mask(self.image[0].shape)(prediction[0])[0]
