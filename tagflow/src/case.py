from typing import Tuple
from pathlib import Path

import numpy as np
from skimage import morphology, measure
from medpy.metric.binary import dc
from sklearn import metrics
import holoviews as hv
from scipy import interpolate

import torch
from torch import nn
import torch.nn.functional as F

from ..src.rbf import RBF, get_principle_strain
from ..src.predict import track
from ..utils import generate_reference


class EvaluationCase():
    
    def __init__(
        self,
        image: np.ndarray = None, video: np.ndarray = None,
        mask: np.ndarray = None, model: nn.Module = None,
        path: str = None, recompute: bool = False,
        target_class: int = 2
    ):
         
        if (path is not None and Path(path).is_file()) and not recompute:
            self.load(path)
        else:
            assert image is not None and video is not None and mask is not None and model is not None
            
            self.image = np.array(image)
            self.video = np.array(video)
            self.mask = np.array(mask)
            
            self.pred = self._segment(model, torch.Tensor(self.image), target_class)
            
            self.deformation_gt = track(
                self.video, self._reference(self.mask), in_st=False
            )
            self.deformation_nn = track(
                self.video, self._reference(self.pred), in_st=False
            )
            
            self.mesh_gt, self.strain_gt = self._strain(self.mask, np.rollaxis(self.deformation_gt, 2))
            self.mesh_nn, self.strain_nn = self._strain(self.pred, np.rollaxis(self.deformation_nn, 2))
            
            if path is not None:
                self.save(path)
        
    def load(self, path: str):
        saved_arrays = np.load(path)
        
        self.image = saved_arrays['image']
        self.video = saved_arrays['video']
        self.mask = saved_arrays['mask']
        self.pred = saved_arrays['pred']
        self.strain_gt = saved_arrays['strain_gt']
        self.strain_nn = saved_arrays['strain_nn']
        self.deformation_gt = saved_arrays['deformation_gt']
        self.deformation_nn = saved_arrays['deformation_nn']
        self.mesh_gt = saved_arrays['mesh_gt']
        self.mesh_nn = saved_arrays['mesh_nn']
        
    def save(self, path: str):
        np.savez(path, image=self.image, video=self.video, mask=self.mask,
                 pred=self.pred,
                 deformation_gt=self.deformation_gt, deformation_nn=self.deformation_nn,
                 mesh_gt=self.mesh_gt, mesh_nn=self.mesh_nn,
                 strain_gt=self.strain_gt, strain_nn=self.strain_nn)
    
    def dice(self) -> float:
        if self.mask is None or self.pred is None:
            raise ValueError('Either ground truth of predicted mask is None.')
        
        assert self.mask.shape == self.pred.shape
        
        return dc(self.pred, self.mask)
        
    def mape(self) -> Tuple[float, float]:
        if self.strain_gt is None or self.strain_nn is None:
            raise ValueError('Either ground truth or predicted strain is None.')
        
        mape_cir = metrics.mean_absolute_percentage_error(
            self.strain_gt.mean(axis=2)[:, 0], self.strain_nn.mean(axis=2)[:, 0])
        mape_rad = metrics.mean_absolute_percentage_error(
            self.strain_gt.mean(axis=2)[:, 1], self.strain_nn.mean(axis=2)[:, 1])

        return mape_cir, mape_rad
    
    def mae(self) -> Tuple[float, float]:
        if self.strain_gt is None or self.strain_nn is None:
            raise ValueError('Either ground truth or predicted strain is None.')
        
        mape_cir = metrics.mean_absolute_error(
            self.strain_gt.mean(axis=2)[:, 0], self.strain_nn.mean(axis=2)[:, 0])
        mape_rad = metrics.mean_absolute_error(
            self.strain_gt.mean(axis=2)[:, 1], self.strain_nn.mean(axis=2)[:, 1])

        return mape_cir, mape_rad

    @staticmethod
    def _segment(model: nn.Module, image: torch.Tensor, target_class: int = 2) -> np.ndarray:

        model.eval()

        inp: torch.Tensor = image.unsqueeze(0).double().clone()
        out: torch.Tensor = model(inp)
        pred: torch.Tensor = F.softmax(out, dim=1).argmax(dim=1).detach()[0]
        pred = (pred == target_class)
        pred = morphology.binary_closing(pred)
        blobs, num = measure.label(pred, background=0, return_num=True)
        sizes = [(blobs == i).sum() for i in range(1, num + 1)]
        blob_index = np.argmax(sizes) + 1

        return (blobs == blob_index)

    @staticmethod
    def _reference(mask: np.ndarray) -> np.ndarray:
                
        points = np.array(np.where(mask)).T
        centre = (points.min(axis=0) + points.max(axis=0)) / 2.
        radius = np.abs(np.linalg.norm(centre - points, axis=1))

        r0 = generate_reference((radius.min(), radius.max(),)) + np.array(centre)

        # Interpolate the mask
        m, n = mask.shape
        x, y = np.linspace(0, n - 1, n), np.linspace(0, m - 1, m)
        p = interpolate.interp2d(x, y, mask, kind='cubic')
        # Select only the reference points present within the mask
        # Meaning that interpolated value at that point should be > .5
        ref0 = r0[np.fromiter((p(xi, yi) for yi, xi in r0), r0.dtype) > .5]
        
        return ref0[:, [1, 0]]
    
    @staticmethod
    def _strain(mask: np.ndarray, deformation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        mesh = np.array(np.where(mask))
        
        # Needs to be (2 x Npoints time) to be solved more efficiently
        deformation = np.swapaxes(deformation, 0, 2)
        
        Nt = deformation.shape[2]
        Np = mesh.shape[1]
        
        rbf = RBF(deformation[:, :, 0], const=12, reg=1e-3)
        
        gl_strain = []

        for time in range(Nt):
            points_t = deformation[:, :, time]
            deformation_grad = np.zeros([Np, 3, 3])
            
            for dim in range(2):
                _ = rbf.solve(points_t[dim, :])
                deformation_grad[:, dim, :2] = rbf.derivative(mesh).T
                
            gl_strain.append(get_principle_strain(mesh, deformation_grad))
        
        return mesh, np.array(gl_strain)

    def visualize(self) -> hv.Layout:
        
        peaks_gt = np.argmax(np.abs(self.strain_gt.mean(axis=2)), axis=0)
        peaks_nn = np.argmax(np.abs(self.strain_nn.mean(axis=2)), axis=0)
                
        times = list(range(25))

        time = hv.Dimension('time', label='Time', unit='s')
        strain_c = hv.Dimension('strain_c', label='Circumferential')
        strain_r = hv.Dimension('strain_c', label='Radial')

        cir_gt = hv.Points((self.mesh_gt[1], 255 - self.mesh_gt[0], self.strain_gt[peaks_gt[0], 0, :]),
                           vdims='strain', group='Peak circumferential strain', label=f'(t={peaks_gt[0]})')
        rad_gt = hv.Points((self.mesh_gt[1], 255 - self.mesh_gt[0], self.strain_gt[peaks_gt[1], 1, :]),
                           vdims='strain', group='Peak radial strain', label=f'(t={peaks_gt[1]})')
            
        cir_nn = hv.Points((self.mesh_nn[1], 255 - self.mesh_nn[0], self.strain_nn[peaks_nn[0], 0, :]),
                           vdims='strain', group='Peak circumferential strain', label=f'(t={peaks_nn[0]})')
        rad_nn = hv.Points((self.mesh_nn[1], 255 - self.mesh_nn[0], self.strain_nn[peaks_nn[1], 1, :]),
                           vdims='strain', group='Peak radial strain', label=f'(t={peaks_nn[1]})')
                        
        strain_lo = (cir_gt + cir_nn + rad_gt + rad_nn).opts(
            hv.opts.Points(color='strain', cmap='viridis', marker='square', size=4, colorbar=True,
                           xaxis=None, yaxis=None)
        )

        cir_t_gt = hv.Curve((times, self.strain_gt.mean(axis=2)[:, 0]), time, strain_c, label='GT',
                            group='Circumferential strain')
        rad_t_gt = hv.Curve((times, self.strain_gt.mean(axis=2)[:, 1]), time, strain_r, label='GT',
                            group='Radial strain')
            
        cir_t_nn = hv.Curve((times, self.strain_nn.mean(axis=2)[:, 0]), time, strain_c, label='NN',
                            group='Circumferential strain')
        rad_t_nn = hv.Curve((times, self.strain_nn.mean(axis=2)[:, 1]), time, strain_r, label='NN',
                            group='Radial strain')
            
        time_lo = (cir_t_gt * cir_t_nn).opts(ylabel=r"$$E_{cc}$$") \
            + (rad_t_gt * rad_t_nn).opts(ylabel=r"$$E_{rr}$$")

        x, y = np.linspace(0, 255, 256), np.linspace(255, 0, 256)
        img = hv.Image((x, y, self.image[0]), label='image').opts(cmap='gray', xaxis=None, yaxis=None)
        gt_mask = hv.Image((x, y, self.mask), label='GT mask').opts(cmap='RdBu', alpha=.4)
        nn_mask = hv.Image((x, y, self.pred), label='NN mask').opts(cmap='RdBu', alpha=.4)

        roi_lo = (img * gt_mask).opts(title='Ground truth segmentation') \
            + (img * nn_mask).opts(title='Predicted segmentation')

        fig = \
            roi_lo.Overlay.I + strain_lo.Peak_circumferential_strain + time_lo.Circumferential_strain \
            + roi_lo.Overlay.II + strain_lo.Peak_radial_strain + time_lo.Radial_strain

        return fig.cols(4)
