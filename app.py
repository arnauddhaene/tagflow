import builtins

import streamlit as st

import tagflow.home
import tagflow.reference
import tagflow.strain

from tagflow.src.predict import predict


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
    
    selected_page = st.sidebar.radio('Go to', PAGES.keys())
    
    st.sidebar.write("""---""")
    
    if st.session_state.reference is None:
        st.sidebar.warning('Tracking reference not set.')
    else:
        if st.session_state.points is None:
            st.sidebar.info('Tracking reference set, ready to compute deformations')
            st.sidebar.button('Launch tracking', on_click=track)
        else:
            st.sidebar.info('Recompute tracking')
            st.sidebar.button('Launch tracking', on_click=track)
    
    PAGES[selected_page].write()


@st.cache(hash_funcs={builtins.complex: lambda _: None})
def track():
    st.session_state.points = predict(st.session_state.image, st.session_state.reference) \
        .swapaxes(0, 2)


if __name__ == '__main__':
    main()
