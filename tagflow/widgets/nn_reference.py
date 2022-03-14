import numpy as np
from skimage import morphology, measure

import torch

from .reference_picker import ReferencePicker
from ..models.segmentation.unet_utils import _preprocess_image, _postprocess_mask
from ..src.predict import segment
from ..utils import generate_reference


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
        out = (out == 2)  # Select MYO class
        out = morphology.binary_closing(out)  # Close segmentation mask
        blobs, num = measure.label(out, background=0, return_num=True)  # Closed components
        sizes = [(blobs == i).sum() for i in range(num)]  # Evaluate component size
        blob_index = np.argmax(sizes) + 1  # Fetch index of largest blob
        
        self.roi = (blobs == blob_index)
        
        points = np.array(np.where(self.roi)).T
        centre = points.mean(axis=0)  # Find center of the mask
        radius = np.abs(np.linalg.norm(centre - points, axis=1))  # Get distance to centre

        # Generate ref points
        r0 = generate_reference((radius.min(), radius.max(),)) + np.array(centre)
        # Mask out the points not in the mask
        ref = np.zeros_like(self.image[0])
        ref[tuple(np.round(r0).astype(int).T.tolist())] = 1
        ref = ref * self.roi  # Intersection with mask
        ref = np.array(np.where(ref)).T
        
        self.ref_points = ref[:, [1, 0]]  # Swap columns
