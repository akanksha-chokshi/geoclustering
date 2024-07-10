import streamlit as st
import os
import folium
import random
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, MultiPolygon
from streamlit_folium import folium_static
import zipfile

def extract_centroid(geometry):
    if isinstance(geometry, MultiPolygon):
        centroid = geometry.centroid
        return centroid
    elif isinstance(geometry, Polygon):
        centroid = geometry.centroid
        return centroid
    else:
        return None

def save_clusters_to_zip(gdf, output_dir):
    zip_filename = "clusters_geojson.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for cluster_id in gdf['cluster'].unique():
            cluster_gdf = gdf[gdf['cluster'] == cluster_id]
            output_file = os.path.join(output_dir, f'cluster_{cluster_id}.geojson')
            cluster_gdf.drop(columns=['centroid', 'cluster']).to_file(output_file, driver='GeoJSON')
            zipf.write(output_file, os.path.basename(output_file))
    return zip_filename

st.title("GeoJSON Clustering App")

# Initialize session state keys
if 'silhouette_scores' not in st.session_state:
    st.session_state['silhouette_scores'] = None
if 'optimal_clusters' not in st.session_state:
    st.session_state['optimal_clusters'] = None
if 'coords' not in st.session_state:
    st.session_state['coords'] = None
if 'gdf' not in st.session_state:
    st.session_state['gdf'] = None
if 'map' not in st.session_state:
    st.session_state['map'] = None
if 'clustering_done' not in st.session_state:
    st.session_state['clustering_done'] = False

uploaded_file = st.file_uploader("Upload your GeoJSON file", type="geojson")

if uploaded_file:
    gdf = gpd.read_file(uploaded_file)

    # Extract centroids
    gdf['centroid'] = gdf['geometry'].apply(extract_centroid)
    coords = gdf['centroid'].apply(lambda x: [x.x, x.y]).tolist()

    start_clusters = st.number_input("Minimum Number of Clusters to Consider", min_value=2, max_value=50, value=10, step=1)
    end_clusters = st.number_input("Maximum Number of Clusters to Consider", min_value=start_clusters, max_value=50, value=50, step=1)

    if st.button("Analyse Best Number of Clusters"):
        silhouette_scores = []

        for k in range(start_clusters, end_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(coords)
            silhouette_avg = silhouette_score(coords, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)

        # Plot silhouette scores
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(start_clusters, end_clusters + 1), silhouette_scores, marker='s', linestyle='-', label='Silhouette Score')
        ax.set_ylabel('Silhouette Score')
        ax.set_xlabel('Number of Clusters')
        ax.legend(loc='lower right')
        plt.grid(True)
        plt.xticks(range(start_clusters, end_clusters + 1))
        plt.tight_layout()
        st.pyplot(fig)

        # Find optimal number of clusters
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + start_clusters
        st.info(f"Suggested Number of Clusters: {optimal_clusters}")
        
        # Store silhouette scores and optimal clusters in session state
        st.session_state['silhouette_scores'] = silhouette_scores
        st.session_state['optimal_clusters'] = optimal_clusters
        st.session_state['coords'] = coords
        st.session_state['gdf'] = gdf

if st.session_state['optimal_clusters'] is not None:
    num_clusters = st.number_input("Number of Clusters", min_value=2, max_value=50, value=st.session_state['optimal_clusters'], step=1)

    if st.button("Perform Clustering"):
        coords = st.session_state['coords']
        gdf = st.session_state['gdf']
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
        cluster_labels = kmeans.labels_
        gdf['cluster'] = cluster_labels

        # Display map
        avg_lat = sum(point.y for point in gdf['centroid']) / len(gdf)
        avg_lon = sum(point.x for point in gdf['centroid']) / len(gdf)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=7)

        clusters = gdf['cluster'].unique()
        colors = {cluster: '#{:06x}'.format(random.randint(0, 0xFFFFFF)) for cluster in clusters}

        for _, row in gdf.iterrows():
            point = row['centroid']
            cluster = row['cluster']
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=5,
                color=colors[cluster],
                fill=True,
                fill_color=colors[cluster]
            ).add_to(m)

        # Save results in session state
        st.session_state['gdf'] = gdf
        st.session_state['map'] = m
        st.session_state['clustering_done'] = True

if st.session_state['clustering_done']:
    gdf = st.session_state['gdf']
    m = st.session_state['map']
    folium_static(m)
    
    output_dir = 'clusters_geojson'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save GeoJSON files for each cluster into a zip
    zip_file = save_clusters_to_zip(gdf, output_dir)

    # Download buttons
    csv_file = "clustered_points.csv"
    gdf.to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        st.download_button("Download Clustered Results as a CSV", f, file_name="clustered_points.csv")

    with open(zip_file, "rb") as f:
        st.download_button("Download Folder Containing All Cluster GeoJSONs", f, file_name="clusters_geojson.zip")
