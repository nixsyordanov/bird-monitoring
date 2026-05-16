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
    page_title="Kalimok Bird Tracking Platform",
    page_icon="🦅",
    layout="wide"
)

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns True if the user entered the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Remove password from memory for security
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First load - show password field
        st.text_input("🔒 Enter password to access the platform:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Wrong password
        st.text_input("🔒 Enter password to access the platform:", type="password", on_change=password_entered, key="password")
        st.error("🚫 Incorrect password. Please try again.")
        return False
    else:
        # Correct password
        return True

# Stop execution if password check fails
if not check_password():
    st.stop()


# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Force dark text color for visibility on white background in both themes */
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] div {
        color: #1f2937 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- DATA LOADING AND PROCESSING ---
@st.cache_data(ttl=600)
def load_and_process_data():
    # Loading data with ';' separator
    df = pd.read_csv("records.csv", sep=';')
    
    # Process time format
    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%d.%m.%Y, %H:%M:%S')
    
    # Filter rows with valid coordinates
    df = df.dropna(subset=['Position'])
    df = df[df['Position'] != 'N/A']
    
    # Split coordinates into Lat and Lon
    df[['Lat', 'Lon']] = df['Position'].str.split(',', expand=True).astype(float)
    
    # Expand data for detailed bio-telemetry charts (5 readings per 48-hour packet)
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


# --- MAIN APP LAYOUT ---
st.title("🦅 Kalimok Bird Tracking Platform")
st.markdown("*Real-time satellite monitoring and behavioral analysis*")
st.markdown("---")

try:
    df_main, df_expanded = load_and_process_data()
    
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2662/2662503.png", width=100)
    st.sidebar.header("Control Panel")
    
    # Transmitter Selection
    available_devices = sorted(df_main['Device'].unique())
    devices = st.sidebar.multiselect(
        "Select Transmitters:", 
        options=available_devices, 
        default=available_devices[:1]
    )
    
    if not devices:
        st.warning("Please select at least one transmitter from the sidebar control panel.")
        st.stop()
    
    # Date Selection
    min_t = df_main['Time (UTC)'].min()
    max_t = df_main['Time (UTC)'].max()
    
    date_range = st.sidebar.date_input(
        "Observation Period:", 
        [min_t.date(), max_t.date()],
        min_value=min_t.date(),
        max_value=max_t.date()
    )
    
    # Handle single date or range selection safely
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0] if isinstance(date_range, list) else date_range

    # Apply Filters to DataFrames
    mask_main = (df_main['Device'].isin(devices)) & \
                 (df_main['Time (UTC)'].dt.date >= start_date) & \
                 (df_main['Time (UTC)'].dt.date <= end_date)
    
    df_filtered = df_main[mask_main].sort_values('Time (UTC)')
    
    mask_exp = (df_expanded['Device'].isin(devices)) & \
               (df_expanded['Time'].dt.date >= start_date) & \
               (df_expanded['Time'].dt.date <= end_date)
    
    df_exp_filtered = df_expanded[mask_exp].sort_values('Time')

    if df_filtered.empty:
        st.info("No data tracking records found for the selected period or devices.")
    else:
        # --- TOP LEVEL METRICS ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Data Points", len(df_filtered))
        with m2:
            st.metric("Active Devices", df_filtered['Device'].nunique())
        with m3:
            st.metric("Latest Sync (UTC)", df_filtered['Time (UTC)'].max().strftime('%H:%M | %d %b'))
        
        st.markdown("---")

        # --- INTERACTIVE TABS ---
        tab1, tab2, tab3 = st.tabs(["📍 Geospatial Map", "📈 Bio-Telemetry", "🎯 Analysis & Clusters"])

        # TAB 1: GEOSPATIAL MAP
        with tab1:
            col_map, col_controls = st.columns([4, 1])
            
            with col_controls:
                st.write("**Map Layers**")
                show_heatmap = st.checkbox("Enable Heatmap view", value=False)
                map_style = st.selectbox("Base Map Style", ["OpenStreetMap", "CartoDB positron"])

            # Center map layout
            m = folium.Map(
                location=[df_filtered['Lat'].mean(), df_filtered['Lon'].mean()], 
                zoom_start=10,
                tiles=map_style
            )
            
            # Palette for distinct transmitter paths
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            device_colors = {dev: colors[i % len(colors)] for i, dev in enumerate(devices)}

            heat_data = []
            for dev in devices:
                dev_df = df_filtered[df_filtered['Device'] == dev]
                if dev_df.empty:
                    continue
                points = dev_df[['Lat', 'Lon']].values.tolist()
                heat_data.extend(points)
                
                # Draw Trajectory lines
                folium.PolyLine(points, color=device_colors[dev], weight=3, opacity=0.8, tooltip=f"Device Path: {dev}").add_to(m)
                
                # Draw solid professional Circle Markers
                for idx, r in dev_df.iterrows():
                    folium.CircleMarker(
                        location=[r['Lat'], r['Lon']],
                        radius=6,
                        color='white',
                        weight=1.5,
                        fill=True,
                        fill_color=device_colors[dev],
                        fill_opacity=1.0,
                        popup=f"<b>Device:</b> {r['Device']}<br><b>Time (UTC):</b> {r['Time (UTC)'].strftime('%Y-%m-%d %H:%M:%S')}",
                        tooltip=f"{r['Device']} | {r['Time (UTC)'].strftime('%d %b, %H:%M')}"
                    ).add_to(m)
            
            if show_heatmap and heat_data:
                HeatMap(heat_data, radius=15, blur=10).add_to(m)
                
            # Static key avoids map reloading/flickering on session changes
            st_folium(m, width="100%", height=600, key="main_map", returned_objects=[])

        # TAB 2: BIO-TELEMETRY CHARTS
        with tab2:
            st.subheader("Biological Activity & Environmental Sensor Trends")
            
            # Activity Line Chart
            fig_vedba = px.line(df_exp_filtered, x='Time', y='Activity (VeDBA)', color='Device', 
                                template="plotly_white", color_discrete_sequence=colors,
                                labels={"Time": "Timeline (UTC)", "Activity (VeDBA)": "VeDBA Activity Level"})
            st.plotly_chart(fig_vedba, use_container_width=True)
            
            # Temperature Line Chart
            fig_temp = px.line(df_exp_filtered, x='Time', y='Temperature (°C)', color='Device', 
                              template="plotly_white", color_discrete_sequence=colors,
                              labels={"Time": "Timeline (UTC)", "Temperature (°C)": "Temperature (°C)"})
            st.plotly_chart(fig_temp, use_container_width=True)

        # TAB 3: DBSCAN CLUSTER ANALYSIS
        with tab3:
            st.subheader("Nesting & Stopover Site Identification")
            st.info("The DBSCAN algorithm automatically Groups spatial coordinates to isolate key behavior hotspots (nesting locations, feeding sites, or prolonged rest areas).")
            
            if len(df_filtered) > 5:
                coords = df_filtered[['Lat', 'Lon']].values
                # 0.005 radians (~500 meters radius detection window)
                db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(coords))
                df_filtered['Cluster'] = db.labels_.astype(str)
                
                clusters_count = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                st.success(f"Algorithm detected **{clusters_count}** significant site hotspots. (Points categorized as '-1' represent transit/noise data).")
                
                fig_cluster = px.scatter_mapbox(
                    df_filtered, lat="Lat", lon="Lon", color="Cluster", 
                    size_max=15, zoom=9, mapbox_style="carto-positron",
                    hover_name="Time (UTC)", title="Identified Spatial Clusters Map"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("Insufficient location data points selected to run spatial cluster calculations.")

except Exception as e:
    st.error(f"An error occurred while tracking configuration or data mapping: {e}")
