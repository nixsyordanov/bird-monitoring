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
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[role="radiogroup"] > label p {
        margin: 0 !important;
        font-size: 1rem !important;
    }
    div[role="radiogroup"] > label:has(input:checked) {
        background-color: rgba(128, 128, 128, 0.25) !important;
        border-color: rgba(128, 128, 128, 0.45) !important;
    }
    div[role="radiogroup"] > label:has(input:checked) p {
        font-weight: 700 !important;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    div[data-testid="stSelectbox"] label, div[data-testid="stToggle"] label {
        display: none !important;
    }
    .map-legend {
        font-size: 0.85rem;
        color: #4b5563;
        background: rgba(128, 128, 128, 0.05);
        padding: 8px 15px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-bottom: 10px;
        display: inline-block;
    }
    .leaflet-control-container .leaflet-left {
        margin-left: 5px;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- EXPERT SUMMARY FUNCTION ---
def generate_point_summary(r):
    vedba_vals = []
    for i in range(1, 6):
        val = r.get(f'VeDBA {i} (raw)')
        try:
            if val != 'N/A' and not pd.isna(val):
                vedba_vals.append(float(val))
        except:
            pass
            
    if not vedba_vals:
        act_text = "неизвестна активност"
    else:
        avg_vedba = sum(vedba_vals) / len(vedba_vals)
        max_vedba = max(vedba_vals)
        
        if avg_vedba > 15 or max_vedba >= 25:
            act_text = "<b>висока активност</b> (вероятен полет или ловуване)"
        elif avg_vedba > 5:
            act_text = "<b>умерена активност</b> (кратки движения)"
        else:
            act_text = "<b>покой</b> (почивка или гнездене)"
            
    radius = r.get('Clean_Radius', 0)
    if radius < 100:
        loc_text = "отлична GPS точност"
    else:
        loc_text = f"приблизителна мрежова локация ({int(radius)} м)"
        
    sig_text = "добра връзка" if str(r.get('LQI')) == 'Good' else "слаб сигнал"
    
    return f"💡 <b>Анализ:</b> Птицата е в състояние на {act_text}, засечена с {loc_text} и {sig_text}."

# --- DATA LOADING ---
@st.cache_data(ttl=600)
def load_and_process_data():
    try:
        df = pd.read_csv("records.csv", sep=';')
        df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%d.%m.%Y, %H:%M:%S')
        df = df.dropna(subset=['Position'])
        df = df[df['Position'] != 'N/A']
        df[['Lat', 'Lon']] = df['Position'].str.split(',', expand=True).astype(float)
        
        def parse_radius(val):
            try:
                return float(str(val).split()[0])
            except:
                return 0
                
        if 'Radius (m) (Source/Status)' in df.columns:
            df['Clean_Radius'] = df['Radius (m) (Source/Status)'].apply(parse_radius)
        else:
            df['Clean_Radius'] = 0
        
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
        
        df_expanded = pd.DataFrame(expanded_list).sort_values(by=['Device', 'Time'])
        
        return df, df_expanded
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

try:
    df_main, df_expanded = load_and_process_data()
    
    # --- SIDEBAR ---
    st.sidebar.header("Controls")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    selected_devices = []
    
    if not df_main.empty:
        latest_times = df_main.groupby('Device')['Time (UTC)'].max().sort_values(ascending=False)
        available_devices = latest_times.index.tolist()
        
        st.sidebar.subheader("Transmitters (Sorted by latest sync)")
        for i, dev in enumerate(available_devices):
            color = colors[i % len(colors)]
            last_sync_formatted = latest_times[dev].strftime('%d %b, %H:%M')
            checkbox_label = f"● {dev} (Last: {last_sync_formatted})"
            
            if st.sidebar.checkbox(checkbox_label, value=(i == 0), key=f"ch_{dev}"):
                selected_devices.append(dev)

        st.sidebar.subheader("Time Range")
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

        nav_col, heat_col, base_col = st.columns([5.5, 1.5, 2.5])
        
        with nav_col:
            active_tab = st.radio("Nav", ["📍 Map View", "📈 Bio-Telemetry", "🎯 Clusters"], horizontal=True, label_visibility="collapsed")
            
        if active_tab == "📍 Map View":
            with heat_col:
                show_heat = st.toggle("🔥 Heatmap", value=False)
            with base_col:
                map_style = st.selectbox("Basemap", ["OpenStreetMap", "Satellite", "CartoDB Positron"], label_visibility="collapsed")
        
        st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)

        if active_tab == "📍 Map View":
            
            st.markdown("""
            <div class="map-legend">
                <b>Legend:</b> 
                🟢 <span style="color: green; font-weight: bold;">Start Point (SP)</span> &nbsp;|&nbsp; 
                🔴 <span style="color: red; font-weight: bold;">End Point (EP)</span> &nbsp;|&nbsp; 
                ⭕ Dashed rings represent the <b>Accuracy Radius</b> of each transmission.
            </div>
            """, unsafe_allow_html=True)

            center_lat, center_lon = df_filtered['Lat'].mean(), df_filtered['Lon'].mean()
            
            if map_style == "Satellite":
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, 
                               tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                               attr='Esri World Imagery')
            elif map_style == "CartoDB Positron":
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')
            else:
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

            Draw(export=False, position='topleft', draw_options={'polyline': True, 'polygon': True, 'circle': False, 'marker': True, 'circlemarker': False}).add_to(m)
            MeasureControl(position='topleft', primary_length_unit='meters', secondary_length_unit='kilometers', primary_area_unit='sqmeters', secondary_area_unit='hectares').add_to(m)
            Fullscreen(position='topleft', title='Expand me', title_cancel='Exit me', force_separate_button=True).add_to(m)

            heat_data = []
            for dev in selected_devices:
                all_dev_data = df_main[df_main['Device'] == dev].sort_values('Time (UTC)')
                points_all = all_dev_data[['Lat', 'Lon']].values.tolist()
                
                dev_df = df_filtered[df_filtered['Device'] == dev].sort_values('Time (UTC)')
                color = colors[available_devices.index(dev) % len(colors)]
                
                if points_all:
                    heat_data.extend(points_all)
                    folium.PolyLine(points_all, color=color, weight=3, opacity=0.6).add_to(m)
                    
                    if not dev_df.empty:
                        sp_row = dev_df.iloc[0]
                        ep_row = dev_df.iloc[-1]
                        
                        folium.Marker(
                            [sp_row['Lat'], sp_row['Lon']], 
                            icon=folium.Icon(color='green', icon='play', prefix='fa'),
                            tooltip=f"Start Point (SP) | {sp_row['Time (UTC)'].strftime('%H:%M %d %b')}"
                        ).add_to(m)
                        
                        folium.Marker(
                            [ep_row['Lat'], ep_row['Lon']], 
                            icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                            tooltip=f"End Point (EP) | {ep_row['Time (UTC)'].strftime('%H:%M %d %b')}"
                        ).add_to(m)

                    for _, r in all_dev_data.iterrows():
                        # Генериране на експертния анализ
                        summary_text = generate_point_summary(r)
                        
                        popup_html = f"""
                        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12px; color: #333333; min-width: 250px; line-height: 1.4;">
                            <h4 style="margin: 0 0 8px 0; color: {color}; font-size: 14px; border-bottom: 2px solid {color}; padding-bottom: 3px;">🛰️ Device: {r['Device']}</h4>
                            
                            <div style="background-color: #f0fdf4; border-left: 3px solid #22c55e; padding: 8px; margin-bottom: 10px; border-radius: 0 4px 4px 0;">
                                {summary_text}
                            </div>
                            
                            <table style="width: 100%; margin-bottom: 8px; font-size: 11px;">
                                <tr><td><b>Time (UTC):</b></td><td style="text-align: right;">{r['Time (UTC)'].strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                                <tr><td><b>Accuracy:</b></td><td style="text-align: right;">{int(r['Clean_Radius'])} m</td></tr>
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
                        
                        # 1. Позицията на птицата (Бялата точка)
                        folium.CircleMarker(
                            [r['Lat'], r['Lon']], 
                            radius=4, color='white', fill=True, fill_color=color, fill_opacity=1, 
                            popup=folium.Popup(popup_html, max_width=320),
                            tooltip=f"{r['Device']} | {r['Time (UTC)'].strftime('%H:%M')}"
                        ).add_to(m)
                        
                        # 2. Радарният пръстен за радиуса (Без фон, само пунктирана линия)
                        if r['Clean_Radius'] > 0:
                            folium.Circle(
                                location=[r['Lat'], r['Lon']],
                                radius=r['Clean_Radius'],
                                color=color,
                                weight=1.5,
                                fill=False, # ТУК МАХАМЕ ЦВЕТНОТО ПЕТНО
                                dash_array='5, 5', # ПУНКТИРАНА ЛИНИЯ
                                interactive=False
                            ).add_to(m)
            
            if show_heat and heat_data:
                HeatMap(heat_data, radius=15, blur=10).add_to(m)

            st_folium(m, width="100%", height=510, key="map_view_main_stable")

        elif active_tab == "📈 Bio-Telemetry":
            df_exp_f = df_expanded[(df_expanded['Device'].isin(selected_devices)) & (df_expanded['Time'].dt.date >= start_d) & (df_expanded['Time'].dt.date <= end_d)].sort_values('Time')
            if not df_exp_f.empty:
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Activity (VeDBA)', color='Device', height=280, markers=True, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)
                st.plotly_chart(px.line(df_exp_f, x='Time', y='Temperature (°C)', color='Device', height=280, markers=True, template="plotly_white", color_discrete_sequence=colors), use_container_width=True)

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
