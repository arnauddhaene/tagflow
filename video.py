import time
import skvideo.io
import skvideo.datasets
import matplotlib.pyplot as plt
import streamlit as st


@st.cache()
def init_state():
    if 'frame' not in st.session_state:
        st.session_state.frame = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False


def toggle_play():
    st.session_state.playing = not st.session_state.playing
    st.session_state.frame -= 1


def reset():
    st.session_state.playing = False
    st.session_state.frame = 0


# Initialize state
init_state()

# Import video in format (T, W, H, C)
video = skvideo.io.vread('test.mp4')
framerate = 60.

# Display
display = plt.imshow(video[st.session_state.frame])
plt.axis('off')

player = st.pyplot(plt)

# Controls
_, lcenter, center, _, _ = st.columns(5)
lcenter.button('Reset', on_click=reset)
center.button('Play/Pause', on_click=toggle_play)

while st.session_state.playing:
    st.session_state.frame = (st.session_state.frame + 1) % video.shape[0]
    time.sleep(1. / framerate)
    display.set_data(video[st.session_state.frame])

    player.pyplot(plt)
