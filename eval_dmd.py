from typing import Iterable
from pathlib import Path
from string import digits

import pandas as pd
import holoviews as hv

import streamlit as st

from tagflow.src.case import EvaluationCase


hv.extension('bokeh')
st.set_page_config(layout='wide')

evals: Iterable[Path] = Path('../dmd_eval').iterdir()
selected_eval: str = st.selectbox('Please select an evaluation folder', [p for p in evals if p.is_dir()])

df = pd.read_csv(str(selected_eval) + '.csv', index_col=0)
# Remove anything after an underscore (_) or any redos (digits within the slice name)
df['slice_corrected'] = df.slice.apply(lambda s: s.split('_')[0].translate(str.maketrans('', '', digits)))

c1, c2, c3 = st.sidebar.columns(3)

c1.metric('Avg. Dice', f'{df.dice.mean() * 100:.2f}')
c2.metric('Avg. MAEc', f'{df.mae_circ.mean():.2f}')
c3.metric('Avg. MAEr', f'{df.mae_radial.mean():.2f}')
c2.metric('Avg. MAPEc', f'{df.mape_circ.mean():.2f}')
c3.metric('Avg. MAPEr', f'{df.mape_radial.mean():.2f}')

scans: Iterable[Path] = Path(selected_eval).iterdir()
selected_scan: str = st.selectbox('Please select a scan to evaluate', scans)

case = EvaluationCase(path=selected_scan)
mae_circ, mae_rad = case.mae()
mape_circ, mape_rad = case.mape()
hd = case.hausdorff_distance()

d, mc, mr, mpc, mpr, hdm = st.columns(6)

d.metric('DICE', f'{case.dice() * 100:.1f}', delta=f'{(case.dice() - df.dice.mean()) * 100:.2f}')
mc.metric('MAEcirc', f'{mae_circ:.2f}', delta=f'{mae_circ - df.mae_circ.mean():.2f}')
mr.metric('MAErad', f'{mae_rad:.2f}', delta=f'{mae_rad - df.mae_radial.mean():.2f}')
mpc.metric('MAPEcirc', f'{mape_circ:.2f}', delta=f'{mape_circ - df.mape_circ.mean():.2f}')
mpr.metric('MAPErad', f'{mape_rad:.2f}', delta=f'{mape_rad - df.mape_radial.mean():.2f}')
hdm.metric('HausD', f'{hd:.2f}')

st.bokeh_chart(hv.render(case.visualize(), backend='bokeh'))
