import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

import cv2

from skimage.filters import gabor_kernel

npz = np.load('sample_data/in_vivo_data.npz')
imt = npz['imt']
r0 = npz['r0']

col1, col2 = st.columns(2)

frequency, theta = \
    col1.slider('Frequency', .01, 1., .1), col2.slider('Theta', 0., 2. * np.pi, np.pi)
sigma = st.slider('Sigma', .5, 5., 1.)

gk1 = np.real(gabor_kernel(frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
gk2 = np.real(gabor_kernel(frequency=frequency, theta=-theta, sigma_x=sigma, sigma_y=sigma))

processed = imt.var(axis=0) / imt.mean(axis=0)
processed = cv2.filter2D(processed, -1, np.ones((5, 5), np.float32) / 5 ** 2)
processed = cv2.filter2D(processed, -1, gk1 / (gk1.shape[0] ** 2))
processed = cv2.filter2D(processed, -1, gk2 / (gk2.shape[0] ** 2))

fig, ax = plt.subplots(1, 2, figsize=(15, 15))

ax[0].imshow(imt.var(axis=0) / imt.mean(axis=0))
ax[1].imshow(processed)

st.pyplot(fig)
