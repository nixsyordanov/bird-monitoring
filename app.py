import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.cluster import DBSCAN
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Kalimok Bird Tracking",
    page_icon="🦅",
    layout="wide"
)

# --- PASSWORD PROTECTION ---
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter password:", type="password", on_change=lambda: st.session_state.update({"password_correct": st.session_state.password == st.secrets["app_password"]}), key="password")
        return False
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- REFINED COMPACT CSS ---
st.markdown("""
    <style>
    /* Нагласяне на основния контейнер */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* Заглавие */
    .main-title {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem !important;
        margin-top: -20px !important;
    }

    /* Стилизиране на балоните (Metrics) */
    div[data-testid="metric-container"] {
        background-color: #f3f4f6 !important; /* Светло сив фон за балона */
        padding: 8px 20px !important;
        border-radius: 50px !important;
        border: 1px solid #d1d5db !important;
        width: fit-content !important;
        margin-bottom: 10px !important;
    }
    
    /* Текст вътре в балоните */
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.75rem !important;
        color: #4b5563 !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1rem !important;
        color: #111827 !important;
        font-weight: bold !important;
    }

    /* Табове */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data(ttl=600)
def load_and_process_data():
    try:
        df = pd.read_csv("records.csv", sep=';')
        df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%d.%m.%Y, %H:%M:%S')
        df = df.dropna(subset=['Position'])
        df = df[df['Position'] != 'N/A']
        df[['Lat', 'Lon']] = df['Position'].str.split(',', expand=True).astype(float)
        
        expanded_list = []
        for _, row in df.iterrows():
            for i in range(1, 6):
                offset_hours = (5 - i) * 9.6
                point_time = row['Time (UTC)'] - timedelta(hours=offset_hours)
                expanded_list.append({
                    'Device': row['Device'], 'Time': point_time,
                    'Activity (VeDBA)': row.get(f'VeDBA {i} (raw)', 0),
                    'Temperature (°C)': row.get(f'Avg. Temp {i} (°C)', 0),
                    'Lat': row['Lat'], 'Lon': row['Lon']
                })
        return df, pd.DataFrame(expanded_list)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

try:
    df_main, df_expanded = load_and_process_data()
    
    # --- SIDEBAR ---
    st.sidebar.header("Controls")
    available_devices = sorted(df_main['Device'].unique()) if not df_main.empty else []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    selected_devices = []
    st.sidebar.subheader("Transmitters")
    for i, dev in enumerate(available_devices):
        color = colors[i % len(colors)]
        if st.sidebar.checkbox(f"● {dev}", value=(i == 0), key=f"ch_{dev}"):
            selected_devices.append(dev)

    st.sidebar.subheader("Time Range")
    if not df_main.empty:
        abs_min, abs_max = df_main['Time (UTC)'].min().date(), df_main['Time (UTC)'].max().date()
        start_def = max(abs_max - timedelta(days=14), abs_min)
        date_range = st.sidebar.date_input("Period:", value=(start_def, abs_max), min_value=abs_min, max_value=abs_max)
    else:
        date_range = []

    # Filtering
    if len(date_range) == 2: start_d, end_d = date_range
    elif len(date_range) == 1: start_d = end_d = date_range[0]
    else: start_d = end_d = None

    if start_d and selected_devices:
        df_filtered = df_main[(df_main['Device'].isin(selected_devices)) & (df_main['Time (UTC)'].dt.date >= start_d) & (df_main['Time (UTC)'].dt.date <= end_d)]
    else:
        df_filtered = pd.DataFrame()

    # --- MAIN UI ---
    st.markdown('<p class="main-title">🦅 Kalimok Bird Tracking Platform</p>', unsafe_allow_html=True)
    
    if not df_filtered.empty:
        # Metrics Row
        col_m = st.columns([0.8, 0.8, 1.2, 3]) 
        col_m[0].metric("Points", len(df_filtered))
        col_m[1].metric("Birds", len(selected_devices))
        col_m[2].metric("Latest Sync", df_filtered['Time (UTC)'].max().strftime('%H:%M | %d %b'))

        tabs = st.tabs(["📍 Map View", "📈 Bio-Telemetry", "🎯 Clusters"])

        with tabs[0]:
            m = folium.Map(location=[df_filtered['Lat'].mean(), df_filtered['Lon'].mean()], zoom_start=10)
            for dev in selected_devices:
                dev_df = df_filtered[df_filtered['Device'] == dev].sort_values('Time (UTC)')
                color = colors[available_devices.index(dev) % len(colors)]
                points = dev_df[['Lat', 'Lon']].values.tolist()
                if points:
                    folium.PolyLine(points, color=color, weight=3).add_to(m)
                    for _, r in dev_df.iterrows():
                        folium.CircleMarker([r['Lat'], r['Lon']], radius=5, color='white', fill=True, fill_color=color, fill_opacity=1, tooltip=f"{r['Device']}").add_to(m)
            st_folium(m, width="100%", height=520, key="map_final")

        with tabs[1]:
            df_exp_f = df_expanded[(df_expanded['Device'].isin(selected_devices)) & (df_expanded['Time'].dt.date >= start_d) & (df_expanded['Time'].dt.date <= end_d)]
            if not df_exp_f.empty:
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Activity (VeDBA)', color='Device', height=280, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Temperature (°C)', color='Device', height=280, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)

        with tabs[2]:
            if len(df_filtered) > 5:
                db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(df_filtered[['Lat', 'Lon']].values))
                df_filtered['Cluster'] = db.labels_.astype(str)
                st.plotly_chart(px.scatter_mapbox(df_filtered, lat="Lat", lon="Lon", color="Cluster", zoom=9, height=500, mapbox_style="carto-positron"), use_container_width=True)
    else:
        st.info("Select transmitters from the sidebar to begin analysis.")

except Exception as e:
    st.error(f"Waiting for configuration... {e}")
