import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.cluster import DBSCAN
from datetime import timedelta, datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Kalimok Bird Tracking Platform",
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

# --- CUSTOM UI/UX CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    div[data-testid="metric-container"] label, div[data-testid="metric-container"] div {
        color: #1f2937 !important;
    }
    .stCheckbox { margin-bottom: -10px; }
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
    
    # --- SIDEBAR: CLEANER SELECTION ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2662/2662503.png", width=80)
    st.sidebar.header("Platform Controls")

    # Transmitter Selection as Checkboxes with Colors
    st.sidebar.subheader("Transmitters")
    available_devices = sorted(df_main['Device'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    selected_devices = []
    for i, dev in enumerate(available_devices):
        color = colors[i % len(colors)]
        # Use a colored dot to link UI with Map
        if st.sidebar.checkbox(f"● {dev}", value=(i == 0), key=f"check_{dev}"):
            selected_devices.append(dev)

    # Date Selection - Default to Last 14 Days
    st.sidebar.subheader("Time Range")
    max_data_date = df_main['Time (UTC)'].max().date()
    default_start_date = max_data_date - timedelta(days=14)
    
    date_range = st.sidebar.date_input(
        "Observation Window:",
        value=(default_start_date, max_data_date),
        min_value=df_main['Time (UTC)'].min().date(),
        max_value=max_data_date
    )

    # Filtering logic
    if len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d = end_d = date_range[0]

    df_filtered = df_main[(df_main['Device'].isin(selected_devices)) & (df_main['Time (UTC)'].dt.date >= start_d) & (df_main['Time (UTC)'].dt.date <= end_d)].sort_values('Time (UTC)')
    df_exp_filtered = df_expanded[(df_expanded['Device'].isin(selected_devices)) & (df_expanded['Time'].dt.date >= start_d) & (df_expanded['Time'].dt.date <= end_d)].sort_values('Time')

    # --- MAIN CONTENT ---
    st.title("🦅 Kalimok Bird Tracking Platform")
    
    if not selected_devices:
        st.info("Please select at least one transmitter from the sidebar to view data.")
    elif df_filtered.empty:
        st.warning("No data found for the selected period.")
    else:
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Locations Found", len(df_filtered))
        m2.metric("Tracked Birds", len(selected_devices))
        m3.metric("Last Update (UTC)", df_filtered['Time (UTC)'].max().strftime('%H:%M | %d %b'))

        tabs = st.tabs(["📍 Movement Map", "📈 Sensors & Activity", "🎯 Hotspot Analysis"])

        with tabs[0]:
            # Map Controls as a clean horizontal expander
            with st.expander("🗺️ Map Display Settings"):
                c1, c2 = st.columns(2)
                show_heat = c1.toggle("Show Density Heatmap", False)
                map_style = c2.selectbox("Style", ["OpenStreetMap", "CartoDB positron"])

            m = folium.Map(location=[df_filtered['Lat'].mean(), df_filtered['Lon'].mean()], zoom_start=10, tiles=map_style)
            
            heat_data = []
            for i, dev in enumerate(selected_devices):
                dev_df = df_filtered[df_filtered['Device'] == dev]
                color = colors[available_devices.index(dev) % len(colors)]
                points = dev_df[['Lat', 'Lon']].values.tolist()
                heat_data.extend(points)
                
                folium.PolyLine(points, color=color, weight=3, opacity=0.8).add_to(m)
                for _, r in dev_df.iterrows():
                    folium.CircleMarker(
                        [r['Lat'], r['Lon']], radius=6, color='white', weight=1,
                        fill=True, fill_color=color, fill_opacity=1,
                        tooltip=f"{r['Device']} | {r['Time (UTC)'].strftime('%H:%M')}"
                    ).add_to(m)
            
            if show_heat: HeatMap(heat_data, radius=15).add_to(m)
            st_folium(m, width="100%", height=600, key="map_stable", returned_objects=[])

        with tabs[1]:
            st.subheader("Bio-Telemetry Trends")
            st.caption("VeDBA (Vectorial Dynamic Body Acceleration) represents the intensity of the bird's movement.")
            st.plotly_chart(px.line(df_exp_filtered, x='Time', y='Activity (VeDBA)', color='Device', template="plotly_white", color_discrete_sequence=colors), use_container_width=True)
            
            st.caption("Ambient temperature recorded by the transmitter sensor.")
            st.plotly_chart(px.line(df_exp_filtered, x='Time', y='Temperature (°C)', color='Device', template="plotly_white", color_discrete_sequence=colors), use_container_width=True)

        with tabs[2]:
            st.subheader("Site Fidelity & Clusters")
            st.write("Identification of frequent stopover or nesting sites based on spatial density.")
            if len(df_filtered) > 5:
                db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(df_filtered[['Lat', 'Lon']].values))
                df_filtered['Cluster'] = db.labels_.astype(str)
                st.plotly_chart(px.scatter_mapbox(df_filtered, lat="Lat", lon="Lon", color="Cluster", zoom=8, mapbox_style="carto-positron"), use_container_width=True)
            else:
                st.warning("Not enough data points for analysis.")

except Exception as e:
    st.error(f"Error loading data: {e}")
