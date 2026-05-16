import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, Draw, MeasureControl, Fullscreen
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

# --- CSS FOR UNIFIED TOOLBAR & PILL RADIO BUTTONS ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* ПРЕВРЪЩАНЕ НА RADIO БУТОНИТЕ В ТАБОВЕ */
    div[role="radiogroup"] {
        flex-direction: row;
        gap: 12px;
        align-items: center;
        margin-bottom: 5px;
    }
    div[role="radiogroup"] > label {
        background-color: rgba(128, 128, 128, 0.08) !important;
        padding: 8px 20px !important;
        border-radius: 30px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out !important;
        margin-bottom: 0 !important;
    }
    div[role="radiogroup"] > label:hover {
        background-color: rgba(128, 128, 128, 0.18) !important;
        border-color: rgba(128, 128, 128, 0.3) !important;
    }
    /* Скриване на оригиналното кръгче на радио бутона */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    /* Стилизиране на текста вътре */
    div[role="radiogroup"] > label p {
        margin: 0 !important;
        font-size: 1rem !important;
    }
    /* Активен (избран) бутон */
    div[role="radiogroup"] > label:has(input:checked) {
        background-color: rgba(128, 128, 128, 0.25) !important;
        border-color: rgba(128, 128, 128, 0.45) !important;
    }
    div[role="radiogroup"] > label:has(input:checked) p {
        font-weight: 700 !important;
    }
    
    /* Вертикално центриране на елементите в колоните */
    div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    /* Скриване на етикетите (labels) на toggle и selectbox, за да са в една линия */
    div[data-testid="stSelectbox"] label, div[data-testid="stToggle"] label {
        display: none !important;
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

        # HEADER (Балони и Емблема)
        header_html = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 5px; margin-bottom: 20px; border-bottom: 1px solid rgba(128, 128, 128, 0.25);">
            <div style="display: flex; gap: 10px; align-items: center;">
                <div style="background-color: rgba(128, 128, 128, 0.12); padding: 5px 14px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); text-align: center; min-width: 75px;">
                    <span style="font-size: 0.65rem; opacity: 0.6; font-weight: 600; display: block; text-transform: uppercase; letter-spacing: 0.5px;">Points</span>
                    <span style="font-size: 0.95rem; font-weight: 700;">{points_count}</span>
                </div>
                <div style="background-color: rgba(128, 128, 128, 0.12); padding: 5px 14px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); text-align: center; min-width: 75px;">
                    <span style="font-size: 0.65rem; opacity: 0.6; font-weight: 600; display: block; text-transform: uppercase; letter-spacing: 0.5px;">Birds</span>
                    <span style="font-size: 0.95rem; font-weight: 700;">{birds_count}</span>
                </div>
                <div style="background-color: rgba(128, 128, 128, 0.12); padding: 5px 14px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2); text-align: center; min-width: 120px;">
                    <span style="font-size: 0.65rem; opacity: 0.6; font-weight: 600; display: block; text-transform: uppercase; letter-spacing: 0.5px;">Latest Sync</span>
                    <span style="font-size: 0.95rem; font-weight: 700;">{latest_sync_str}</span>
                </div>
            </div>
            <div style="font-size: 1.3rem; font-weight: 700; letter-spacing: -0.5px; white-space: nowrap; opacity: 0.9;">
                🦅 Kalimok Bird Tracking Platform
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

        # --- UNIFIED TOOLBAR ROW ---
        # Създаваме колони: една широка за навигацията и две тесни за контролите
        nav_col, heat_col, base_col = st.columns([5.5, 1.5, 2.5])
        
        with nav_col:
            active_tab = st.radio("Nav", ["📍 Map View", "📈 Bio-Telemetry", "🎯 Clusters"], horizontal=True, label_visibility="collapsed")
            
        if active_tab == "📍 Map View":
            with heat_col:
                show_heat = st.toggle("🔥 Heatmap", value=False)
            with base_col:
                # OpenStreetMap вече е на първо място, съответно се зарежда по подразбиране
                map_style = st.selectbox("Basemap", ["OpenStreetMap", "Satellite", "CartoDB Positron"], label_visibility="collapsed")
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

        # --- CONTENT RENDER BASED ON SELECTION ---
        if active_tab == "📍 Map View":
            center_lat, center_lon = df_filtered['Lat'].mean(), df_filtered['Lon'].mean()
            
            # Избор на подложка според селекцията
            if map_style == "Satellite":
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, 
                               tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                               attr='Esri World Imagery')
            elif map_style == "CartoDB Positron":
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')
            else:
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

            # Плъгини за чертане, измерване и цял екран
            Draw(export=True, position='topleft', draw_options={'polyline': True, 'polygon': True, 'circle': False, 'marker': True, 'circlemarker': False}).add_to(m)
            MeasureControl(position='topright', primary_length_unit='meters', secondary_length_unit='kilometers', primary_area_unit='sqmeters', secondary_area_unit='hectares').add_to(m)
            Fullscreen(position='topright', title='Expand me', title_cancel='Exit me', force_separate_button=True).add_to(m)

            heat_data = []
            for dev in selected_devices:
                dev_df = df_filtered[df_filtered['Device'] == dev].sort_values('Time (UTC)')
                color = colors[available_devices.index(dev) % len(colors)]
                points = dev_df[['Lat', 'Lon']].values.tolist()
                
                if points:
                    heat_data.extend(points)
                    folium.PolyLine(points, color=color, weight=3).add_to(m)
                    
                    for _, r in dev_df.iterrows():
                        popup_html = f"""
                        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12px; color: #333333; min-width: 240px; line-height: 1.4;">
                            <h4 style="margin: 0 0 5px 0; color: {color}; font-size: 14px; border-bottom: 2px solid {color}; padding-bottom: 3px;">🛰️ {r['Device']}</h4>
                            <table style="width: 100%; margin-bottom: 8px; font-size: 11px;">
                                <tr><td><b>Time (UTC):</b></td><td style="text-align: right;">{r['Time (UTC)'].strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                                <tr><td><b>Coords:</b></td><td style="text-align: right;">{r['Lat']:.5f}, {r['Lon']:.5f}</td></tr>
                                <tr><td><b>Seq No:</b></td><td style="text-align: right;">{int(r.get('Sequence Number', 0))}</td></tr>
                                <tr><td><b>Signal:</b></td><td style="text-align: right;">{r.get('LQI', 'N/A')}</td></tr>
                            </table>
                            <div style="font-weight: bold; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; color: #555; border-top: 1px solid #ddd; padding-top: 6px;">📊 Packet Burst Readings</div>
                            <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 11px; border: 1px solid #dddddd;">
                                <tr style="background-color: #f9f9f9; font-weight: bold; border-bottom: 1px solid #ddd;">
                                    <th style="padding: 3px; border-right: 1px solid #ddd;">#</th>
                                    <th style="padding: 3px; border-right: 1px solid #ddd;">VeDBA</th>
                                    <th style="padding: 3px;">Temp</th>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;"><td>1</td><td style="border-right: 1px solid #eee;">{r.get('VeDBA 1 (raw)', 'N/A')}</td><td>{r.get('Avg. Temp 1 (°C)', 'N/A')}°C</td></tr>
                                <tr style="border-bottom: 1px solid #eee;"><td>2</td><td style="border-right: 1px solid #eee;">{r.get('VeDBA 2 (raw)', 'N/A')}</td><td>{r.get('Avg. Temp 2 (°C)', 'N/A')}°C</td></tr>
                                <tr style="border-bottom: 1px solid #eee;"><td>3</td><td style="border-right: 1px solid #eee;">{r.get('VeDBA 3 (raw)', 'N/A')}</td><td>{r.get('Avg. Temp 3 (°C)', 'N/A')}°C</td></tr>
                                <tr style="border-bottom: 1px solid #eee;"><td>4</td><td style="border-right: 1px solid #eee;">{r.get('VeDBA 4 (raw)', 'N/A')}</td><td>{r.get('Avg. Temp 4 (°C)', 'N/A')}°C</td></tr>
                                <tr><td>5</td><td style="border-right: 1px solid #eee;">{r.get('VeDBA 5 (raw)', 'N/A')}</td><td>{r.get('Avg. Temp 5 (°C)', 'N/A')}°C</td></tr>
                            </table>
                        </div>
                        """
                        folium.CircleMarker([r['Lat'], r['Lon']], radius=5, color='white', fill=True, fill_color=color, fill_opacity=1, popup=folium.Popup(popup_html, max_width=300)).add_to(m)
            
            if show_heat and heat_data:
                HeatMap(heat_data, radius=15, blur=10).add_to(m)

            st_folium(m, width="100%", height=510, key="map_view_main")

        elif active_tab == "📈 Bio-Telemetry":
            df_exp_f = df_expanded[(df_expanded['Device'].isin(selected_devices)) & (df_expanded['Time'].dt.date >= start_d) & (df_expanded['Time'].dt.date <= end_d)]
            if not df_exp_f.empty:
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Activity (VeDBA)', color='Device', height=280, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Temperature (°C)', color='Device', height=280, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)

        elif active_tab == "🎯 Clusters":
            if len(df_filtered) > 5:
                db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(df_filtered[['Lat', 'Lon']].values))
                df_filtered['Cluster'] = db.labels_.astype(str)
                st.plotly_chart(px.scatter_mapbox(df_filtered, lat="Lat", lon="Lon", color="Cluster", zoom=9, height=520, mapbox_style="carto-positron"), use_container_width=True)
            else:
                st.warning("Not enough data points to run clustering algorithm.")
                
    else:
        st.info("Select transmitters from the sidebar to begin analysis.")

except Exception as e:
    st.error(f"Error: {e}")
