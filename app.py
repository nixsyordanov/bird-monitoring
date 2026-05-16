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

# --- COMPACT UX/UI CSS ---
st.markdown("""
    <style>
    /* Премахване на излишните разстояния най-отгоре */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    h1 {
        margin-top: -30px !important;
        padding-bottom: 10px !important;
        font-size: 2rem !important;
    }
    .stMarkdown p {
        margin-bottom: 5px !important;
    }
    
    /* Стилизиране на метриките като малки балони */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 5px 15px !important;
        border-radius: 20px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        min-width: fit-content !important;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.8rem !important;
        color: #6b7280 !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.1rem !important;
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    /* Свиване на разстоянието между табовете и съдържанието */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data(ttl=600)
def load_and_process_data():
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

try:
    df_main, df_expanded = load_and_process_data()
    
    # --- SIDEBAR ---
    st.sidebar.header("Controls")
    available_devices = sorted(df_main['Device'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    selected_devices = []
    st.sidebar.subheader("Transmitters")
    for i, dev in enumerate(available_devices):
        color = colors[i % len(colors)]
        if st.sidebar.checkbox(f"● {dev}", value=(i == 0), key=f"ch_{dev}"):
            selected_devices.append(dev)

    st.sidebar.subheader("Time Range")
    abs_min, abs_max = df_main['Time (UTC)'].min().date(), df_main['Time (UTC)'].max().date()
    start_def = max(abs_max - timedelta(days=14), abs_min)
    
    date_range = st.sidebar.date_input("Period:", value=(start_def, abs_max), min_value=abs_min, max_value=abs_max)

    # Filter
    if len(date_range) == 2: start_d, end_d = date_range
    else: start_d = end_d = date_range[0]

    df_filtered = df_main[(df_main['Device'].isin(selected_devices)) & (df_main['Time (UTC)'].dt.date >= start_d) & (df_main['Time (UTC)'].dt.date <= end_d)]

    # --- MAIN UI ---
    st.title("🦅 Kalimok Bird Tracking Platform")
    
    if selected_devices and not df_filtered.empty:
        # Компактни метрики в един ред
        col_m = st.columns([1, 1, 1.5, 4]) # Последната колона е празна, за да избута балоните вляво
        col_m[0].metric("Points", len(df_filtered))
        col_m[1].metric("Birds", len(selected_devices))
        col_m[2].metric("Latest Sync", df_filtered['Time (UTC)'].max().strftime('%H:%M | %d %b'))

        tabs = st.tabs(["📍 Map", "📈 Sensors", "🎯 Clusters"])

        with tabs[0]:
            # Картата е с малко по-малка височина (500), за да се вижда всичко без скрол
            m = folium.Map(location=[df_filtered['Lat'].mean(), df_filtered['Lon'].mean()], zoom_start=10)
            
            heat_data = []
            for dev in selected_devices:
                dev_df = df_filtered[df_filtered['Device'] == dev].sort_values('Time (UTC)')
                color = colors[available_devices.index(dev) % len(colors)]
                points = dev_df[['Lat', 'Lon']].values.tolist()
                heat_data.extend(points)
                folium.PolyLine(points, color=color, weight=3).add_to(m)
                for _, r in dev_df.iterrows():
                    folium.CircleMarker([r['Lat'], r['Lon']], radius=5, color='white', fill=True, fill_color=color, fill_opacity=1, tooltip=f"{r['Device']}").add_to(m)
            
            st_folium(m, width="100%", height=500, key="map_v3")

        with tabs[1]:
            df_exp_f = df_expanded[(df_expanded['Device'].isin(selected_devices)) & (df_expanded['Time'].dt.date >= start_d) & (df_expanded['Time'].dt.date <= end_d)]
            st.plotly_chart(px.line(df_exp_f, x='Time', y='Activity (VeDBA)', color='Device', height=300, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)
            st.plotly_chart(px.line(df_exp_f, x='Time', y='Temperature (°C)', color='Device', height=300, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)

        with tabs[2]:
            if len(df_filtered) > 5:
                db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(df_filtered[['Lat', 'Lon']].values))
                df_filtered['Cluster'] = db.labels_.astype(str)
                st.plotly_chart(px.scatter_mapbox(df_filtered, lat="Lat", lon="Lon", color="Cluster", zoom=9, height=500, mapbox_style="carto-positron"), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
