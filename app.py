import builtins

import numpy as np

import streamlit as st

import tagflow.home
import tagflow.reference
import tagflow.strain

from tagflow.src.predict import track
from tagflow.state.state import SessionState, SessionStatus


STANFORD_LOGO = 'https://disruptingafrica.com/images/4/42/Stanford_Logo.jpg'

st.set_page_config(layout="centered", page_title="tagflow", page_icon=STANFORD_LOGO)

PAGES = {
    'Home': tagflow.home,
    'Reference': tagflow.reference,
    'Strain': tagflow.strain
}


def main():
    
    tagflow.home.init()
    
    # Handle unexpected behavior that I don't like at all
    if 'reference' not in st.session_state:
        with st.spinner('App is being restarted and cache is being cleared.'):
            st.legacy_caching.clear_cache()
            st.experimental_rerun()
    
    st.sidebar.title('tagflow Navigation')
    st.sidebar.write('Automated myocardial strain estimation using tagged MR images')
    
    selected_page = st.sidebar.radio('Go to', PAGES.keys(), key='page')
    
    st.sidebar.write("""---""")
    
    ss = SessionState()
    
    if ss.status().value < SessionStatus.reference.value:
        st.sidebar.warning('Tracking reference not set.')
    else:
        if ss.status() == SessionStatus.reference:
            st.sidebar.info('Tracking reference set, ready to compute deformations')
        else:
            st.sidebar.info('Recompute tracking')
        st.sidebar.button('Launch tracking', on_click=_track, args=(ss.reference.value(),))
        
    PAGES[selected_page].write()


@st.cache(hash_funcs={builtins.complex: lambda _: None})
def _track(reference: np.ndarray):
    # Receives reference so that Streamlit knows to rerun _track when reference changes
    ss = SessionState()
    deformation = track(ss.image.value(), reference)
    ss.deformation.update(np.rollaxis(deformation, 2))
    # Switch to home page once deformation has been predicted
    st.session_state.page = 'Home'


if __name__ == '__main__':
    main()
