import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance

import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.write('Testing drawable canvas')


@st.cache
def load_data():
    npz = np.load('torch_track/sample_data/in_vivo_data.npz')
    y1r = np.load('torch_track/sample_data/in_vivo_data_deformation.npz')['y1r']
    imt, r0 = npz['imt'], npz['r0']
    return imt, r0, y1r.swapaxes(0, 2)


imt, _, y1r = load_data()
r0 = None

xdim, ydim = tuple(imt.shape[1:])

xmin, xmax = (xdim * .25, xdim * .75)
ymin, ymax = (ydim * .25, ydim * .75)

contrast = st.sidebar.slider('Contrast enhancement', .5, 5., 1.25)
brightness = st.sidebar.slider('Brightness enhancement', .5, 5., 1.25)

drawing_mode = st.sidebar.selectbox("Drawing tool:", ("circle", "transform"))

image = Image.fromarray(imt[0, int(xmin):int(xmax), int(ymin):int(ymax)])\
    .convert(mode='RGB')

image = ImageEnhance.Contrast(image).enhance(contrast)
image = ImageEnhance.Brightness(image).enhance(brightness)

canvas_result = st_canvas(
    fill_color='#FF0000',
    stroke_width=3.,
    stroke_color='#FF0000',
    background_image=image,
    update_streamlit=True,
    height=5 * (int(xmax) - int(xmin)), width=5 * (int(ymax) - int(ymin)),
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.json_data is not None:
    # need to convert obj to str because PyArrow
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    st.write(objects)
    

if st.sidebar.button('Save reference frame'):
    
    tl = np.array(objects[['left', 'top']])
    wh = np.array(objects[['width', 'height']])
    
    r0 = np.array([ymin, xmin]) + ((tl - wh / 2.)) / 5.
    
if r0 is not None:
    fig, ax = plt.subplots(1, figsize=(12, 8))

    ax.imshow(imt[0], cmap='gray')
    # ax.axis('off')
    
    plt.gca().add_patch(plt.Rectangle((ymin, xmin), ymax - ymin, xmax - xmin, fill=False))
    ax.scatter(r0[:, 0], r0[:, 1], 30, c='r', marker='x')
    
    st.pyplot(plt)
