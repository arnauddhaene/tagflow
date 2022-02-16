import requests

import streamlit as st

import tagflow.home
import tagflow.reference
import tagflow.strain


STANFORD_LOGO = 'https://disruptingafrica.com/images/4/42/Stanford_Logo.jpg'

st.set_page_config(layout="centered", page_title="tagflow", page_icon=STANFORD_LOGO)

PAGES = {
    'Home': tagflow.home,
    'Reference': tagflow.reference,
    'Strain': tagflow.strain
}


@st.cache
def init():
    
    storage = ['points', 'reference', 'roi', 'image']
    
    for item in storage:
        if item not in st.session_state:
            st.session_state[item] = None


def main():
    
    init()
    
    st.title('tagflow')
    st.write('Automated myocardial strain estimation using tagged MR images')
    
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', PAGES.keys())
    
    st.sidebar.write("""---""")
    
    if st.session_state.reference is None:
        st.sidebar.warning('Tracking reference not set.')
    else:
        if st.session_state.points is None:
            st.sidebar.info('Tracking reference set, ready to compute deformations')
            st.sidebar.button('Launch tracking', on_click=track)
    
    PAGES[selected_page].write()


@st.cache
def track():
        
    payload = {
        'images': st.session_state.image.tolist(),
        'points': st.session_state.reference.tolist()
    }
    
    track_result = requests.post('http://127.0.0.1:5000/track', json=payload).json()

    y1r = track_result['prediction'] + st.session_state.reference[:, :, None]
    st.session_state.points = y1r.swapaxes(0, 2)
    

if __name__ == '__main__':
    main()
