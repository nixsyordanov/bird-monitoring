import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.cluster import DBSCAN
from datetime import timedelta

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="WildCloud Nanofox Анализ", layout="wide")

@st.cache_data(ttl=600)
def load_and_process_data():
    # Зареждане (използваме ';' като разделител)
    df = pd.read_csv("records.csv", sep=';')
    
    # Пречистване на времето
    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%d.%m.%Y, %H:%M:%S')
    
    # Филтриране само на редове с координати
    df = df.dropna(subset=['Position'])
    df = df[df['Position'] != 'N/A']
    
    # Разделяне на координатите
    df[['Lat', 'Lon']] = df['Position'].str.split(',', expand=True).astype(float)
    
    # --- РАЗГРЪЩАНЕ НА ДАННИТЕ ЗА ГРАФИКИ ---
    # Тъй като имаме 5 измервания на пакет (за 48 часа), създаваме нови записи
    expanded_list = []
    for _, row in df.iterrows():
        for i in range(1, 6):
            # Времеви интервал: приблизително на всеки 9.6 часа назад от часа на предаване
            offset_hours = (5 - i) * 9.6
            point_time = row['Time (UTC)'] - timedelta(hours=offset_hours)
            
            expanded_list.append({
                'Device': row['Device'],
                'Time': point_time,
                'VeDBA': row.get(f'VeDBA {i} (raw)', 0),
                'Temp': row.get(f'Avg. Temp {i} (°C)', 0),
                'Lat': row['Lat'],
                'Lon': row['Lon']
            })
    
    return df, pd.DataFrame(expanded_list)

# --- ПРИЛОЖЕНИЕ ---
st.title("🦅 Анализ на данни: Nanofox 1g")

try:
    df_main, df_expanded = load_and_process_data()
    
    # --- SIDEBAR ФИЛТРИ ---
    st.sidebar.header("Филтри")
    devices = st.sidebar.multiselect("Избери устройства:", df_main['Device'].unique(), default=df_main['Device'].unique()[:1])
    
    # Филтър по време
    min_t = df_main['Time (UTC)'].min()
    max_t = df_main['Time (UTC)'].max()
    date_range = st.sidebar.date_input("Период:", [min_t.date(), max_t.date()])
    
    # Прилагане на филтри
    mask = (df_main['Device'].isin(devices)) & \
           (df_main['Time (UTC)'].dt.date >= date_range[0]) & \
           (df_main['Time (UTC)'].dt.date <= date_range[1])
    
    df_filtered = df_main[mask].sort_values('Time (UTC)')
    df_exp_filtered = df_expanded[df_expanded['Device'].isin(devices)]
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    tab1, tab2, tab3 = st.tabs(["📍 Карта и Траектории", "📊 Активност и Температура", "🧠 Клъстери (Места за почивка)"])

    with tab1:
        st.subheader("Движение на птиците")
        m = folium.Map(location=[df_filtered['Lat'].mean(), df_filtered['Lon'].mean()], zoom_start=8)
        
        # Слоеве
        heat_data = []
        for dev in devices:
            dev_df = df_filtered[df_filtered['Device'] == dev]
            points = dev_df[['Lat', 'Lon']].values.tolist()
            heat_data.extend(points)
            
            # Траектория
            folium.PolyLine(points, color=np.random.choice(['blue', 'red', 'green', 'purple']), weight=2, opacity=0.7).add_to(m)
            for idx, r in dev_df.iterrows():
                folium.CircleMarker([r['Lat'], r['Lon']], radius=4, popup=f"{r['Device']} @ {r['Time (UTC)']}", fill=True).add_to(m)
        
        # Heatmap опция
        show_heatmap = st.checkbox("Покажи Heatmap (гъстота на позициите)")
        if show_heatmap:
            HeatMap(heat_data).add_to(m)
            
        st_folium(m, width=1000, height=500)

    with tab2:
        st.subheader("Динамика на показателите")
        # Графика Активност
        fig_vedba = px.line(df_exp_filtered, x='Time', y='VeDBA', color='Device', title="Активност (VeDBA)")
        st.plotly_chart(fig_vedba, use_container_width=True)
        
        # Графика Температура
        fig_temp = px.line(df_exp_filtered, x='Time', y='Temp', color='Device', title="Температура (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab3:
        st.subheader("Автоматично идентифициране на ключови места")
        st.info("Използваме алгоритъма DBSCAN за откриване на групи точки, където птицата е прекарала повече време.")
        
        if len(df_filtered) > 3:
            coords = df_filtered[['Lat', 'Lon']].values
            # kms_per_radian = 6371.0088
            # epsilon = 0.5 / kms_per_radian (търсим точки в радиус 500 метра)
            db = DBSCAN(eps=0.005, min_samples=3).fit(np.radians(coords))
            df_filtered['Cluster'] = db.labels_
            
            clusters_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            st.write(f"Открити потенциални зони (гнезда/почивка): {clusters_found}")
            
            fig_cluster = px.scatter_mapbox(df_filtered, lat="Lat", lon="Lon", color="Cluster", 
                                          hover_name="Time (UTC)", mapbox_style="carto-positron", zoom=7)
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("Няма достатъчно данни за клъстериране.")

except Exception as e:
    st.error(f"Моля, качете 'records.csv' в основната папка. Грешка: {e}")