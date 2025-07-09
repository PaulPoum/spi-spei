# Streamlit app for computing and visualizing SPI & SPEI
# Dependencies: streamlit, pandas, numpy, scipy, plotly, matplotlib

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gamma, logistic, norm
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="SPI & SPEI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 SPI & SPEI Interactive Dashboard")
st.markdown(
    "Cette application calcule et visualise dynamiquement le SPI et le SPEI pour différentes stations à partir d’un fichier Excel."
)

# Sidebar: file upload and parameters
st.sidebar.header("Paramètres")
uploaded_file = st.sidebar.file_uploader(
    "Charger le classeur Excel (.xlsx)", type=["xlsx"]
)

if not uploaded_file:
    st.warning("Veuillez charger un fichier Excel avec les feuilles 'station' et 'data'.")
    st.stop()

# Read data
xls = pd.ExcelFile(uploaded_file)
df_station = pd.read_excel(xls, sheet_name="station")
df_data = pd.read_excel(xls, sheet_name="data", parse_dates=["date"])

# Validate
required_station = {"station_id","name","latitude","longitude","altitude","region","country"}
required_data = {"station_id","date","temp_max","temp_min","temp_moy","precipitation","evapo"}
if not required_station.issubset(df_station.columns):
    st.error("Feuille 'station' : colonnes manquantes.")
    st.stop()
if not required_data.issubset(df_data.columns):
    st.error("Feuille 'data' : colonnes manquantes.")
    st.stop()

# Merge
df = df_data.merge(df_station, on="station_id", how="left")

# Sidebar selections
station_name = st.sidebar.selectbox(
    "Sélectionner une station", df_station['name'].unique()
)
scale = st.sidebar.slider("Échelle (mois)", 1, 24, 12)

# Filter for chosen station
df_loc = df[df['name']==station_name].sort_values('date').set_index('date')
if df_loc.empty:
    st.error(f"Aucune donnée pour {station_name}")
    st.stop()

# Compute rolling sums
precip = df_loc['precipitation'].rolling(scale).sum().dropna()
balance = (df_loc['precipitation'] - df_loc['evapo']).rolling(scale).sum().dropna()

# SPI: handle zeros
p = precip.values
n_zero = (p==0).sum(); p_zero = n_zero/len(p)
pos = p[p>0]
params = gamma.fit(pos, floc=0)
cdf_p = np.where(p==0, p_zero, p_zero + (1-p_zero)*gamma.cdf(p,*params))
cdf_p = np.clip(cdf_p,1e-6,1-1e-6)
spi = pd.Series(norm.ppf(cdf_p), index=precip.index)

# SPEI: logistic
b = balance.values
loc0, scale0 = logistic.fit(b)
cdf_b = logistic.cdf(b, loc=loc0, scale=scale0)
cdf_b = np.clip(cdf_b,1e-6,1-1e-6)
spei = pd.Series(norm.ppf(cdf_b), index=balance.index)

# Layout: two columns for charts
col1, col2 = st.columns(2)

# Helper to separate positive/negative
pos_spi = spi.clip(lower=0)
neg_spi = spi.clip(upper=0)
pos_spei = spei.clip(lower=0)
neg_spei = spei.clip(upper=0)

with col1:
    st.subheader(f"SPI-{scale} : {station_name}")
    fig_spi = go.Figure()
    fig_spi.add_trace(go.Scatter(
        x=pos_spi.index, y=pos_spi, fill='tozeroy', mode='none', name='Humide', fillcolor='rgba(0, 116, 217, 0.6)'
    ))
    fig_spi.add_trace(go.Scatter(
        x=neg_spi.index, y=neg_spi, fill='tozeroy', mode='none', name='Sec', fillcolor='rgba(255, 65, 54, 0.6)'
    ))
    fig_spi.update_layout(
        yaxis_title='SPI', xaxis_title='Année', template='plotly_white', legend=dict(y=0.99, x=0.01)
    )
    st.plotly_chart(fig_spi, use_container_width=True)

with col2:
    st.subheader(f"SPEI-{scale} : {station_name}")
    fig_spei = go.Figure()
    fig_spei.add_trace(go.Scatter(
        x=pos_spei.index, y=pos_spei, fill='tozeroy', mode='none', name='Humide', fillcolor='rgba(46, 204, 113, 0.6)'
    ))
    fig_spei.add_trace(go.Scatter(
        x=neg_spei.index, y=neg_spei, fill='tozeroy', mode='none', name='Sec', fillcolor='rgba(255, 65, 54, 0.6)'
    ))
    fig_spei.update_layout(
        yaxis_title='SPEI', xaxis_title='Année', template='plotly_white', legend=dict(y=0.99, x=0.01)
    )
    st.plotly_chart(fig_spei, use_container_width=True)

# Export data
st.sidebar.markdown("---")
export_df = pd.DataFrame({
    'Date': spi.index.date,
    f'SPI_{scale}m': spi.values,
    f'SPEI_{scale}m': spei.values
})
csv = export_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label='📥 Exporter SPI & SPEI en CSV',
    data=csv,
    file_name=f'{station_name}_SPI_SPEI_{scale}mois.csv',
    mime='text/csv'
)

# Interpretation & Analysis
st.markdown("---")
col3, col4 = st.columns(2)
with col3:
    st.subheader("Dernières valeurs & interprétation")
    latest = spi.index.max().strftime('%Y-%m')
    spi_val = spi.iloc[-1]; spei_val = spei.iloc[-1]
    def interp(v):
        if v<=-2: return 'Sécheresse extrême'
        if v<=-1.5: return 'Sécheresse sévère'
        if v<=-1: return 'Sécheresse modérée'
        if v<1: return 'Normal'
        if v<1.5: return 'Humide modéré'
        if v<2: return 'Très humide'
        return 'Humide extrême'
    st.metric(f"SPI {latest}", f"{spi_val:.2f}", interp(spi_val))
    st.metric(f"SPEI {latest}", f"{spei_val:.2f}", interp(spei_val))

with col4:
    st.subheader("Statistiques clés SPI")
    stats = spi.describe().to_frame().T
    stats.columns = [c.title() for c in stats.columns]
    st.table(stats)
    drought = (spi< -1).sum(); wet = (spi>1).sum()
    st.write(f"# sécheresse modérée+: {drought}")
    st.write(f"# humidité modérée+: {wet}")

# Expanders: analyses détaillées
with st.expander("🔍 Analyse détaillée SPI"):
    st.write("- **Durée maximale de sécheresse** : ", (spi<0).astype(int).groupby((spi>=0).astype(int).cumsum()).sum().max(), "mois")
    st.write("- **Période la plus critique** : ", spi.idxmin().strftime('%Y-%m'), f"(SPI = {spi.min():.2f})")

with st.expander("🔍 Analyse détaillée SPEI"):
    st.write("- **Durée maximale de déficit** : ", (spei<0).astype(int).groupby((spei>=0).astype(int).cumsum()).sum().max(), "mois")
    st.write("- **Période la plus critique** : ", spei.idxmin().strftime('%Y-%m'), f"(SPEI = {spei.min():.2f})")

st.markdown("---")
st.markdown("*Note : SPI & SPEI sont des indices normalisés – 0 = valeur moyenne, positif = plus humide, négatif = plus sec.*")
