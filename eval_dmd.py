from typing import Iterable
from pathlib import Path

import holoviews as hv

import streamlit as st

from tagflow.src.case import EvaluationCase


hv.extension('bokeh')
st.set_page_config(layout='wide')

evals: Iterable[Path] = Path('../dmd_eval').iterdir()
selected_eval: str = st.selectbox('Please select an evaluation folder', [p for p in evals if p.is_dir()])

scans: Iterable[Path] = Path(selected_eval).iterdir()
selected_scan: str = st.selectbox('Please select a scan to evaluate', scans)

case = EvaluationCase(path=selected_scan)
mae_circ, mae_rad = case.mae()
mape_circ, mape_rad = case.mape()

d, mc, mr, mpc, mpr = st.columns(5)

d.metric('DICE', f'{case.dice() * 100:.1f}')
mc.metric('MAEcirc', f'{mae_circ:.2f}')
mr.metric('MAErad', f'{mae_rad:.2f}')
mpc.metric('MAPEcirc', f'{mape_circ:.2f}')
mpr.metric('MAPErad', f'{mape_rad:.2f}')

st.bokeh_chart(hv.render(case.visualize(), backend='bokeh'))
