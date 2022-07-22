from pathlib import Path

import numpy as np

import h5py

import pydicom
import nibabel as nib

import click


@click.command()
@click.option('--infile', help='Filename of input Nifti file or DICOM folder.')
@click.option('--outfile', help='Filename of output HDF5 file.')
def convert(infile: str, outfile: str) -> None:
    """Conversion script from Nifti or DICOM to HDF5"""

    if Path(infile).suffixes == ['.nii', '.gz']:
        array = nib.load(infile).get_data()
    elif Path(infile).is_dir():
        array = []
        files = [f for f in Path(infile).iterdir() if f.suffix == '.dcm']
        for t in sorted(files):
            array.append(pydicom.dcmread(t).pixel_array)
        
        array = np.array(array)
    else:
        raise ValueError('This script supports only Nifti and DICOM images.')

    print(f'Array of shape {array.shape} was extracted from {Path(infile).name}.')

    if Path(outfile).suffix != '.h5':
        outfile = outfile + '.h5'

    hf = h5py.File(outfile, 'w')
    hf.create_dataset('sequence', data=array)
    hf.close()

    print(f'Array saved in {Path(outfile).name}.')


if __name__ == '__main__':
    convert()
