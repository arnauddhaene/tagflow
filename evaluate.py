from pathlib import Path
from tqdm import tqdm

import pandas as pd

import click

import torch
from torch import nn

from tagflow.src.case import EvaluationCase
from tagflow.models.segmentation.unet import UNet
from tagflow.data.datasets import DMDTimeDataset


@click.command()
@click.option('--name', default='dmd_eval', help="Folder name for saving evaluation files.")
@click.option('--model-name', default='model_cine_tag_only_myo_v0_finetuned_dmd_v3.pt', help="Model name")
def run(name, model_name):
    
    model_path = Path('tagflow/network_saves') / model_name
    model: nn.Module = UNet(n_channels=1, n_classes=2, bilinear=True).double()
    # Load old saved version of the model as a state dictionary
    saved_model_sd = torch.load(model_path, map_location=torch.device('cpu'))
    # Extract UNet if saved model is parallelized
    model.load_state_dict(saved_model_sd)
    
    dataset = DMDTimeDataset('../dmd_roi/')
    
    Path(f'../dmd_eval/{name}').mkdir()
    
    storage = []

    for idx, (image, video, mask) in tqdm(enumerate(dataset), total=len(dataset), unit='scan'):
        slic = dataset.slices[idx]
        
        ec = EvaluationCase(image, video, mask, model, recompute=True, target_class=1,
                            path=f'../dmd_eval/{name}/scan_{idx}_{slic}.npz')
        
        mape_circ, mape_radial = ec.mape()
        mae_circ, mae_radial = ec.mae()
        
        storage.append(dict(slice=slic, dice=ec.dice(), mape_circ=mape_circ, mape_radial=mape_radial,
                            mae_circ=mae_circ, mae_radial=mae_radial))

    df = pd.DataFrame(storage)

    df.to_csv(f'../dmd_eval/{name}.csv')


if __name__ == '__main__':
    run()
