from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

import click

from monai.networks import nets

import torch
from torch import nn

from tagflow.src.case import EvaluationCase
# from tagflow.models.segmentation.unet import UNet
from tagflow.data.datasets import DMDTimeDataset


@click.command()
@click.option('--name', default='dmd_eval', help="Folder name for saving evaluation files.")
@click.option('--model-name', default='model_cine_tag_only_myo_v4_finetuned_dmd_v0.pt', help="Model name")
def run(name, model_name):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = Path('tagflow/network_saves') / model_name
    model: nn.Module = nets.SegResNetVAE(
        in_channels=1, out_channels=2,
        input_image_size=(256, 256), spatial_dims=2
    ).double()
    # Load old saved version of the model as a state dictionary
    saved_model_sd = torch.load(model_path, map_location=device)
    # Extract UNet if saved model is parallelized
    model.load_state_dict(saved_model_sd)
    model.to(device)
    
    dataset = DMDTimeDataset('../dmd_roi/')
    
    Path(f'../dmd_eval/{name}').mkdir()
    
    storage = []

    for idx, (image, video, mask) in tqdm(enumerate(dataset), total=len(dataset), unit='scan'):
        slic = dataset.slices[idx]
        
        ec = EvaluationCase(image, video, mask, model, recompute=True, target_class=1,
                            path=f'../dmd_eval/{name}/scan_{idx}_{slic}.npz')
        
        if np.sum(ec.pred) == 0:
            continue
        
        mape_circ, mape_radial = ec.mape()
        mae_circ, mae_radial = ec.mae()
        hd = ec.hausdorff_distance()
        
        storage.append(dict(slice=slic, dice=ec.dice(), mape_circ=mape_circ, mape_radial=mape_radial,
                            mae_circ=mae_circ, mae_radial=mae_radial, hd=hd))

    df = pd.DataFrame(storage)

    df.to_csv(f'../dmd_eval/{name}.csv')


if __name__ == '__main__':
    run()
