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

# --- COMPACT PAGE PADDING & THEME ADAPTATION ---
st.markdown("""
    <style>
    /* Осигуряваме място под навигационната лента на Streamlit, за да няма изрязване */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 0rem !important;
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
    if not df_filtered.empty:
        points_count = len(df_filtered)
        birds_count = len(selected_devices)
        latest_sync_str = df_filtered['Time (UTC)'].max().strftime('%H:%M | %d %b')

        # НАПЪЛНО АДАПТИВЕН ХЕДЪР: Без бял фон, с интелигентни полупрозрачни балони
        header_html = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 5px; margin-bottom: 20px; border-bottom: 1px solid rgba(128, 128, 128, 0.25);">
            <div style="display: flex; gap: 10px; align-items: center;">
                <div style="background-color: rgba(128, 128, 128, 0.12); padding: 5px 14px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); text-align: center; min-width: 75px;">
                    <span style="font-size: 0.65rem; opacity: 0.6; font-weight: 600; display: block; text-transform: uppercase; letter-spacing: 0.5px; color: inherit;">Points</span>
                    <span style="font-size: 0.95rem; font-weight: 700; color: inherit;">{points_count}</span>
                </div>
                <div style="background-color: rgba(128, 128, 128, 0.12); padding: 5px 14px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); text-align: center; min-width: 75px;">
                    <span style="font-size: 0.65rem; opacity: 0.6; font-weight: 600; display: block; text-transform: uppercase; letter-spacing: 0.5px; color: inherit;">Birds</span>
                    <span style="font-size: 0.95rem; font-weight: 700; color: inherit;">{birds_count}</span>
                </div>
                <div style="background-color: rgba(128, 128, 128, 0.12); padding: 5px 14px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); text-align: center; min-width: 120px;">
                    <span style="font-size: 0.65rem; opacity: 0.6; font-weight: 600; display: block; text-transform: uppercase; letter-spacing: 0.5px; color: inherit;">Latest Sync</span>
                    <span style="font-size: 0.95rem; font-weight: 700; color: inherit;">{latest_sync_str}</span>
                </div>
            </div>
            <div style="font-size: 1.3rem; font-weight: 700; color: inherit; letter-spacing: -0.5px; white-space: nowrap; opacity: 0.9;">
                🦅 Kalimok Bird Tracking Platform
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

        # ТАБОВЕ И СЪДЪРЖАНИЕ
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
            st_folium(m, width="100%", height=510, key="map_final_v5")

        with tabs[1]:
            df_exp_f = df_expanded[(df_expanded['Device'].isin(selected_devices)) & (df_expanded['Time'].dt.date >= start_d) & (df_expanded['Time'].dt.date <= end_d)]
            if not df_exp_f.empty:
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Activity (VeDBA)', color='Device', height=250, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Temperature (°C)', color='Device', height=250, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)

        with tabs[2]:
            if len(df_filtered) > 5:
                db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(df_filtered[['Lat', 'Lon']].values))
                df_filtered['Cluster'] = db.labels_.astype(str)
                st.plotly_chart(px.scatter_mapbox(df_filtered, lat="Lat", lon="Lon", color="Cluster", zoom=9, height=500, mapbox_style="carto-positron"), use_container_width=True)
    else:
        st.info("Select transmitters from the sidebar to begin analysis.")

except Exception as e:
    st.error(f"Error: {e}")
