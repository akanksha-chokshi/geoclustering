import streamlit as st
import os
import folium
import random
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from shapely.geometry import Polygon, MultiPolygon
from streamlit_folium import folium_static
import zipfile
from shapely.geometry import shape

# Function to extract centroid
def extract_centroid(geometry):
    if isinstance(geometry, MultiPolygon):
        centroid = geometry.centroid
        return centroid
    elif isinstance(geometry, Polygon):
        centroid = geometry.centroid
        return centroid
    else:
        return None

# Function to simplify geometries
def simplify_geometries(gdf, tolerance=0.001):
    gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
    return gdf

# Function to save clusters to a zip file
def save_clusters_to_zip(gdf, output_dir, simplify, tolerance=0.001):
    zip_filename = "clusters_geojson.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for cluster_id in gdf['cluster'].unique():
            cluster_gdf = gdf[gdf['cluster'] == cluster_id]
            if simplify:
                cluster_gdf = simplify_geometries(cluster_gdf, tolerance)
            output_file = os.path.join(output_dir, f'cluster_{cluster_id}.geojson')
            cluster_gdf.drop(columns=['centroid', 'cluster']).to_file(output_file, driver='GeoJSON')
            zipf.write(output_file, os.path.basename(output_file))
    return zip_filename

# Initialize Streamlit app
st.title("GeoJSON Clustering App")

# Initialize session state keys
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None
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

# Reset session state if a new file is uploaded
if uploaded_file and (uploaded_file != st.session_state['last_uploaded_file']):
    st.session_state.clear()
    st.session_state['last_uploaded_file'] = uploaded_file
    st.session_state['optimal_clusters'] = None
    st.experimental_rerun()
    
if uploaded_file:
    gdf = gpd.read_file(uploaded_file)
    
    # Calculate file size in MB
    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)

    # Determine the minimum and default number of clusters based on file size
    min_clusters = max(2, int(file_size_mb) + 1)
    default_clusters = min_clusters

    # Extract centroids
    gdf['centroid'] = gdf['geometry'].apply(extract_centroid)
    coords = gdf['centroid'].apply(lambda x: [x.x, x.y]).tolist()

    start_clusters = st.number_input("Minimum Number of Clusters to Consider", min_value=2, max_value=50, value=min_clusters, step=1)
    end_clusters = st.number_input("Maximum Number of Clusters to Consider", min_value=start_clusters, max_value=50, value=50, step=1)

    if st.button("Analyse Best Number of Clusters"):
        silhouette_scores = []
        random_state = 0

        for k in range(start_clusters, end_clusters + 1):
            agg = AgglomerativeClustering(n_clusters=k)
            agg.fit(coords)
            silhouette_avg = silhouette_score(coords, agg.labels_)
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
else:
    st.session_state['optimal_clusters'] = None
    st.session_state['clustering_done'] = None

if st.session_state['optimal_clusters'] is not None:
    num_clusters = st.number_input("Number of Clusters", min_value=2, max_value=50, value=st.session_state['optimal_clusters'], step=1)
    st.session_state['clustering_done'] = None

    if st.button("Perform Clustering"):
        coords = st.session_state['coords']
        gdf = st.session_state['gdf']
        random_state = 0

        agg = AgglomerativeClustering(n_clusters=num_clusters).fit(coords)
        cluster_labels = agg.labels_
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
    
    csv_file = "clustered_points.csv"
    gdf.to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        st.download_button("Download Clustered Results as CSV", f, file_name="clustered_points.csv")

    original_output_dir = 'original_clusters_geojson'
    os.makedirs(original_output_dir, exist_ok=True)
    original_zip_file = save_clusters_to_zip(gdf, original_output_dir, simplify=False)
    
    with open(original_zip_file, "rb") as f:
        st.download_button("Download Original Cluster GeoJSONs as ZIP", f, file_name="original_clusters_geojson.zip")

    simplified_output_dir = 'simplified_clusters_geojson'
    os.makedirs(simplified_output_dir, exist_ok=True)
    simplified_zip_file = save_clusters_to_zip(gdf, simplified_output_dir, simplify=True)

    # Download buttons
    
    with open(simplified_zip_file, "rb") as f:
        st.download_button("Download Simplified Cluster GeoJSONs as ZIP", f, file_name="simplified_clusters_geojson.zip")
