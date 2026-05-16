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

# Custom CSS for a more professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_view_proxy=True)

@st.cache_data(ttl=600)
def load_and_process_data():
    # Loading data with ';' separator
    df = pd.read_csv("records.csv", sep=';')
    
    # Process time
    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%d.%m.%Y, %H:%M:%S')
    
    # Filter rows with coordinates
    df = df.dropna(subset=['Position'])
    df = df[df['Position'] != 'N/A']
    
    # Split coordinates
    df[['Lat', 'Lon']] = df['Position'].str.split(',', expand=True).astype(float)
    
    # Expand data for detailed charts
    expanded_list = []
    for _, row in df.iterrows():
        for i in range(1, 6):
            offset_hours = (5 - i) * 9.6
            point_time = row['Time (UTC)'] - timedelta(hours=offset_hours)
            
            expanded_list.append({
                'Device': row['Device'],
                'Time': point_time,
                'Activity (VeDBA)': row.get(f'VeDBA {i} (raw)', 0),
                'Temperature (°C)': row.get(f'Avg. Temp {i} (°C)', 0),
                'Lat': row['Lat'],
                'Lon': row['Lon']
            })
    
    return df, pd.DataFrame(expanded_list)

# --- APP LAYOUT ---
st.title("🦅 Kalimok Bird Tracking Platform")
st.markdown("*Real-time satellite monitoring and behavioral analysis*")

try:
    df_main, df_expanded = load_and_process_data()
    
    # --- SIDEBAR ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2662/2662503.png", width=100)
    st.sidebar.header("Control Panel")
    
    devices = st.sidebar.multiselect(
        "Select Transmitters:", 
        options=sorted(df_main['Device'].unique()), 
        default=df_main['Device'].unique()[:1]
    )
    
    min_t = df_main['Time (UTC)'].min()
    max_t = df_main['Time (UTC)'].max()
    
    date_range = st.sidebar.date_input(
        "Observation Period:", 
        [min_t.date(), max_t.date()],
        min_value=min_t.date(),
        max_value=max_t.date()
    )
    
    # Filters
    mask = (df_main['Device'].isin(devices)) & \
           (df_main['Time (UTC)'].dt.date >= date_range[0]) & \
           (df_main['Time (UTC)'].dt.date <= date_range[1])
    
    df_filtered = df_main[mask].sort_values('Time (UTC)')
    df_exp_filtered = df_expanded[df_expanded['Device'].isin(devices)]

    # --- METRICS ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Data Points", len(df_filtered))
    with m2:
        st.metric("Active Devices", df_filtered['Device'].nunique())
    with m3:
        st.metric("Latest Sync (UTC)", df_filtered['Time (UTC)'].max().strftime('%H:%M | %d %b'))

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📍 Geospatial Map", "📈 Bio-Telemetry", "🎯 Analysis & Clusters"])

    with tab1:
        col_map, col_controls = st.columns([4, 1])
        
        with col_controls:
            st.write("Map Settings")
            show_heatmap = st.checkbox("Enable Heatmap", value=False)
            show_labels = st.checkbox("Show Time Labels", value=True)
            map_style = st.selectbox("Base Map", ["OpenStreetMap", "Stamen Terrain", "CartoDB positron"])

        # Map creation
        m = folium.Map(
            location=[df_filtered['Lat'].mean(), df_filtered['Lon'].mean()], 
            zoom_start=10,
            tiles=map_style
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        device_colors = {dev: colors[i % len(colors)] for i, dev in enumerate(devices)}

        heat_data = []
        for dev in devices:
            dev_df = df_filtered[df_filtered['Device'] == dev]
            points = dev_df[['Lat', 'Lon']].values.tolist()
            heat_data.extend(points)
            
            # Trajectory
            folium.PolyLine(points, color=device_colors[dev], weight=3, opacity=0.8, tooltip=f"Path: {dev}").add_to(m)
            
            # Markers
            for idx, r in dev_df.iterrows():
                folium.CircleMarker(
                    location=[r['Lat'], r['Lon']],
                    radius=6,
                    color='white',
                    weight=1,
                    fill=True,
                    fill_color=device_colors[dev],
                    fill_opacity=0.9,
                    popup=f"<b>Device:</b> {r['Device']}<br><b>Time:</b> {r['Time (UTC)']}",
                    tooltip=f"{r['Device']} @ {r['Time (UTC)'].strftime('%H:%M')}"
                ).add_to(m)
        
        if show_heatmap:
            HeatMap(heat_data, radius=15).add_to(m)
            
        st_folium(m, width="100%", height=600, key="main_map")

    with tab2:
        st.subheader("Biological Activity & Environment")
        fig_vedba = px.line(df_exp_filtered, x='Time', y='Activity (VeDBA)', color='Device', 
                            template="plotly_white", color_discrete_sequence=colors)
        st.plotly_chart(fig_vedba, use_container_width=True)
        
        fig_temp = px.area(df_exp_filtered, x='Time', y='Temperature (°C)', color='Device', 
                          template="plotly_white", color_discrete_sequence=colors)
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab3:
        st.subheader("Nesting & Stopover Site Identification")
        st.info("Using DBSCAN algorithm to find clusters where the bird spends significant time.")
        
        if len(df_filtered) > 5:
            coords = df_filtered[['Lat', 'Lon']].values
            db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(coords))
            df_filtered['Cluster'] = db.labels_.astype(str)
            
            fig_cluster = px.scatter_mapbox(
                df_filtered, lat="Lat", lon="Lon", color="Cluster", 
                size_max=15, zoom=8, mapbox_style="carto-positron",
                title="Identified Clusters (Noise = -1)"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("Insufficient data points for cluster analysis.")

except Exception as e:
    st.error(f"Waiting for data or configuration error. Details: {e}")
