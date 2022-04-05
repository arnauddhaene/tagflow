from pathlib import Path
from typing import Tuple

import pydicom

import numpy as np
from numpy import ndarray
import nibabel as nib
from skimage.draw import polygon, polygon2mask


INDEX_TO_CLASS = dict(zip(range(4), ['BG', 'RV', 'MYO', 'LV']))
CLASS_TO_INDEX = {v: k for k, v in INDEX_TO_CLASS.items()}


class Patient():
    """Class that loads cine MR images and annotations from an ACDC patient.
    """
    
    def __init__(self, filepath: str):
        
        # Fetch list of all potential files
        files = [f for f in Path(filepath).iterdir() if f.suffixes == ['.nii', '.gz']]
        
        for f in files:
            # Discard 4d
            # Discard ground truth as those are fetched by their image
            if '_4d' in str(f) or '_gt' in str(f):
                continue
            
            # Fetch path of mask following dataset nomenclature
            f_gt = self._gt_path(f)
            
            if f_gt in files:
                self.images, self.masks = self.fetch_frames(f, f_gt)
                      
    @staticmethod
    def _gt_path(filepath: Path) -> Path:
        """Get an image's corresponding ground truth mask

        Args:
            filepath (Path): location of image.

        Returns:
            Path: location of mask.
        """
        return filepath.parent / (filepath.stem.split('.')[0] + '_gt.nii.gz')
            
    @staticmethod
    def fetch_frames(image_path: Path, mask_path: Path) -> Tuple[ndarray, ndarray]:
        """Load data from image and mask locations

        Args:
            image_path (Path): location of image.
            mask_path (Path): location of mask.

        Returns:
            Tuple[ndarray, ndarray]: image (1, H, W) and mask (1, H, W)
        """
        
        imt, _, _ = load_nii(image_path)
        gt, _, _ = load_nii(mask_path)
        
        return imt.swapaxes(0, 2), gt.swapaxes(0, 2)
        

class Slice:
    
    def __init__(self, path: Path):
        
        self.slice = path.name
        
        dcm_images = [f for f in path.iterdir() if f.is_file() and f.suffix == '.dcm']
        if len(dcm_images) > 0:
            res = map(lambda ds: (ds.InstanceNumber, ds.pixel_array), map(pydicom.dcmread, dcm_images))
            self.image = np.array(list(zip(*sorted(res, key=lambda item: item[0])))[1])
        
        # Extract ROI if it exists
        roi_path = (path / 'roi_pts.npz')
        self.slice_location = path.name
        if roi_path.is_file():
            rois = np.load(roi_path)
            self.roi = {array_key: rois[array_key] for array_key in list(rois.keys())}
            self.mask = {}
            for _roi in ['pts_interp_outer', 'pts_interp_inner']:
                if self.roi[_roi] is not None:
                    # Verify dimensions
                    assert self.roi[_roi].shape[0] == 2
                    
                    pg = polygon(self.roi[_roi][1, :], self.roi[_roi][0, :], self.image.shape[1:])
                    # Save under 'outer' or 'inner' key
                    self.mask[_roi.split('_')[-1]] = polygon2mask(self.image.shape[1:], np.array(pg).T)
                else:
                    self.mask[_roi.split('_')[-1]] = None
            
        else:
            self.roi = None
            
    def is_annotated(self) -> bool:
        return self.roi is not None
            
    def __repr__(self) -> str:
        return self.__str__()
            
    def __str__(self) -> str:
        return f"{self.slice} slice with{'out' if self.roi is None else ''} annotated ROI."
        

class Scan:
    
    def __init__(self, path: Path, label: bool = False):
        
        self.id = path.name
        self.label = label
        self.slices = {s.name: Slice(s) for s in (path / 'clean').iterdir() if s.is_dir()}


#
# Utils function to load and save nifti files with the nibabel package
#
def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header