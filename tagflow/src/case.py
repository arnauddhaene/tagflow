from typing import Tuple
from pathlib import Path

import copy

import numpy as np
from skimage import morphology, measure
from medpy.metric.binary import dc
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import holoviews as hv
# from scipy import interpolate
from monai.metrics import compute_hausdorff_distance

import torch
from torch import nn
import torch.nn.functional as F

from ..src.rbf import RBF, get_principle_strain
from ..src.predict import track
from ..utils import detect_peaks


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
            
            if np.sum(self.pred) == 0:
                print('No mask predicted for case.')
            else:
                self.deformation_gt = track(
                    self.video, self._reference(self.mask)
                )
                self.deformation_nn = track(
                    self.video, self._reference(self.pred)
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
    
    def hausdorff_distance(self) -> float:
        if self.mask is None or self.pred is None:
            raise ValueError('Either ground truth of predicted mask is None.')
        
        assert self.mask.shape == self.pred.shape
        
        return compute_hausdorff_distance(
            self.mask[None, None, :, :],
            self.pred[None, None, :, :]
        ).item()
        
    def mape(self) -> Tuple[float, float]:
        if self.strain_gt is None or self.strain_nn is None:
            raise ValueError('Either ground truth or predicted strain is None.')
        
        mape_cir = mean_absolute_percentage_error(
            self.strain_gt.mean(axis=2)[:, 0], self.strain_nn.mean(axis=2)[:, 0])
        mape_rad = mean_absolute_percentage_error(
            self.strain_gt.mean(axis=2)[:, 1], self.strain_nn.mean(axis=2)[:, 1])

        return mape_cir, mape_rad
    
    def mae(self) -> Tuple[float, float]:
        if self.strain_gt is None or self.strain_nn is None:
            raise ValueError('Either ground truth or predicted strain is None.')
        
        mape_cir = mean_absolute_error(
            self.strain_gt.mean(axis=2)[:, 0], self.strain_nn.mean(axis=2)[:, 0])
        mape_rad = mean_absolute_error(
            self.strain_gt.mean(axis=2)[:, 1], self.strain_nn.mean(axis=2)[:, 1])

        return mape_cir, mape_rad

    @staticmethod
    def _segment(model: nn.Module, image: torch.Tensor, target_class: int = 2) -> np.ndarray:

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inp: torch.Tensor = image.unsqueeze(0).double().clone().to(device)
        out = model(inp)[0]
        pred: torch.Tensor = F.softmax(out, dim=1).argmax(dim=1).detach().cpu()[0]
        pred = (pred == target_class)
        
        pred = morphology.binary_closing(pred)
        blobs, num = measure.label(pred, background=0, return_num=True)
        sizes = [(blobs == i).sum() for i in range(1, num + 1)]
        if len(sizes) > 0:
            blob_index = np.argmax(sizes) + 1
            return (blobs == blob_index)
        return pred

    @staticmethod
    def _reference(mask: np.ndarray, method: str, image: np.ndarray = None) -> np.ndarray:

        points = np.array(np.where(mask)).T

        if len(points) > 0:
            if method == 'intersections':
                assert image is not None, 'I need the image, fool.'

                z = copy.deepcopy(image)
                xs, ys = np.where(mask == 0)
                z[xs, ys] = np.nan

                peaks = detect_peaks(-z)

                return np.array(np.where(peaks.T)).T

            elif method == 'mesh':
                r0, landmarks = [], [[], []]

                contours = measure.find_contours(mask)
                contours = list(map(lambda c: c[:, ::-1], contours))
                contours[0] = contours[0][::-1]
                stops = np.array(
                    list(map(lambda pts: np.linspace(0, len(pts), 24, endpoint=False), contours)),
                    dtype=np.int16
                )
                
                for idxs in stops.T:
                    for c, i in enumerate(idxs):
                        landmarks[c].append(contours[c][i])

                landmarks = np.array(landmarks)

                for l_id in range(landmarks.shape[1]):
                    points = tuple(map(lambda axe: np.linspace(*axe, 9), landmarks[:, l_id, :].T))
                    r0.extend(np.array(points)[:, 1:-1].T)

                return np.array(r0)

            else:
                raise ValueError('Do better. Give me intersections or mesh')
            
            # centre = (points.min(axis=0) + points.max(axis=0)) / 2.
            # radius = np.abs(np.linalg.norm(centre - points, axis=1))

            # r0 = generate_reference((radius.min(), radius.max(),)) + np.array(centre)

            # # Interpolate the mask
            # m, n = mask.shape
            # x, y = np.linspace(0, n - 1, n), np.linspace(0, m - 1, m)
            # p = interpolate.interp2d(x, y, mask, kind='cubic')
            # # Select only the reference points present within the mask
            # # Meaning that interpolated value at that point should be > .5
            # ref0 = r0[np.fromiter((p(xi, yi) for yi, xi in r0), r0.dtype) > .5]
            
            # return ref0[:, [1, 0]]

        else:
            return np.empty((1, 2))

    @staticmethod
    def _strain(mask: np.ndarray, deformation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        mesh = np.array(np.where(mask.T)).T
        center = mesh.mean(axis=0)
        mesh = (mesh - center).T

        # Needs to be (2 x Npoints x time)
        deformation = np.swapaxes(deformation, 0, 2)
        deformation = deformation - center[:, None, None]

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
        
        strain = np.array(gl_strain)
        # strain[np.isnan(strain)] = 0.
        
        return (mesh.T + center).T, strain

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
