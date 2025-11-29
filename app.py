"""
Geospatial Clustering & Business Location Recommendation System
Streamlit Dashboard Application

This dashboard provides an interactive interface for:
- Selecting city and business type
- Configuring analysis parameters
- Viewing POI maps and clusters
- Exploring heatmaps and recommendations
- Downloading analysis results
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium, folium_static
import time

from modules.fetch_data import DataFetcher
from modules.clean_data import DataCleaner
from modules.clustering import GeoClusterer
from modules.visualize import MapVisualizer
from modules.scoring import LocationScorer


os.makedirs('data', exist_ok=True)
os.makedirs('maps', exist_ok=True)


st.set_page_config(
    page_title="Business Location Recommender",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'cleaned_pois' not in st.session_state:
        st.session_state.cleaned_pois = None
    if 'clustered_pois' not in st.session_state:
        st.session_state.clustered_pois = None
    if 'top_locations' not in st.session_state:
        st.session_state.top_locations = None
    if 'center' not in st.session_state:
        st.session_state.center = None
    if 'cluster_stats' not in st.session_state:
        st.session_state.cluster_stats = None


def run_analysis_pipeline(city: str, business_type: str, radius_km: float,
                          weights: dict, progress_bar) -> dict:
    """
    Run the complete analysis pipeline with progress updates.
    
    Args:
        city: City name
        business_type: Business type to analyze
        radius_km: Search radius in km
        weights: Scoring weights dictionary
        progress_bar: Streamlit progress bar object
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    progress_bar.progress(10, "Fetching city location...")
    fetcher = DataFetcher(city, radius_km)
    center = fetcher.get_city_center()
    
    if center is None:
        st.error(f"Could not find city: {city}. Please check the spelling.")
        return None
    
    results['center'] = center
    st.session_state.center = center
    
    progress_bar.progress(20, "Fetching POI data from OpenStreetMap...")
    
    categories = [
        'shop', 'restaurant', 'cafe', 'supermarket', 'bank',
        'hospital', 'pharmacy', 'school', 'hotel', 'fuel',
        'parking', 'bus_station', business_type
    ]
    categories = list(set(categories))
    
    all_pois = fetcher.fetch_all_pois(categories)
    
    if len(all_pois) == 0:
        st.error("No POIs found in the specified area. Try a larger radius or different city.")
        return None
    
    fetcher.save_to_csv(all_pois, 'raw_data.csv')
    results['raw_poi_count'] = len(all_pois)
    
    progress_bar.progress(40, "Cleaning and preparing data...")
    cleaner = DataCleaner()
    cleaned_pois = cleaner.clean_geodataframe(all_pois)
    
    fetcher.save_to_csv(cleaned_pois, 'cleaned_data.csv')
    st.session_state.cleaned_pois = cleaned_pois
    results['cleaned_poi_count'] = len(cleaned_pois)
    results['cleaning_report'] = cleaner.get_cleaning_report()
    
    progress_bar.progress(55, "Performing clustering analysis...")
    coords = cleaner.prepare_for_clustering(cleaned_pois)
    
    clusterer = GeoClusterer()
    eps_km = min(radius_km / 10, 0.5)
    min_samples = max(3, int(len(coords) / 100))
    
    labels = clusterer.dbscan_clustering(coords, eps_km=eps_km, min_samples=min_samples)
    hotspots = clusterer.detect_hotspots(coords)
    sparse_regions = clusterer.detect_sparse_regions(coords, center[0], center[1])
    
    clustered_pois = clusterer.add_cluster_labels(cleaned_pois)
    clusterer.save_clusters(clustered_pois)
    
    st.session_state.clustered_pois = clustered_pois
    st.session_state.cluster_stats = clusterer.cluster_stats
    results['cluster_stats'] = clusterer.cluster_stats
    results['hotspots'] = hotspots
    results['sparse_regions'] = sparse_regions
    
    progress_bar.progress(70, "Scoring potential locations...")
    scorer = LocationScorer(weights)
    
    supporting_categories = fetcher.get_supporting_pois(business_type)
    candidates = scorer.generate_candidate_locations(
        center[0], center[1],
        radius_km=radius_km,
        grid_size=15
    )
    
    scores = scorer.score_locations(
        candidates,
        cleaned_pois,
        business_type,
        supporting_categories
    )
    
    scorer.save_scores()
    top_locations = scorer.get_top_locations(10)
    st.session_state.top_locations = top_locations
    results['top_locations'] = top_locations
    results['scoring_report'] = scorer.get_analysis_report()
    
    progress_bar.progress(85, "Generating maps...")
    viz = MapVisualizer(center[0], center[1], zoom_start=13)
    
    try:
        all_pois_map = viz.create_all_pois_map(cleaned_pois)
        viz.save_map(all_pois_map, 'all_pois_map.html')
    except Exception as e:
        st.warning(f"Could not generate POI map: {e}")
    
    try:
        cluster_map = viz.create_cluster_map(clustered_pois, clusterer.cluster_stats)
        viz.save_map(cluster_map, 'cluster_map.html')
    except Exception as e:
        st.warning(f"Could not generate cluster map: {e}")
    
    try:
        heatmap = viz.create_heatmap(cleaned_pois)
        viz.save_map(heatmap, 'heatmap.html')
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")
    
    try:
        competition_map = viz.create_competition_heatmap(cleaned_pois, business_type)
        viz.save_map(competition_map, 'competition_heatmap.html')
    except Exception as e:
        st.warning(f"Could not generate competition heatmap: {e}")
    
    try:
        recommendations_map = viz.create_recommendations_map(
            cleaned_pois, top_locations, business_type
        )
        viz.save_map(recommendations_map, 'recommendations_map.html')
    except Exception as e:
        st.warning(f"Could not generate recommendations map: {e}")
    
    progress_bar.progress(100, "Analysis complete!")
    time.sleep(0.5)
    
    return results


def display_metrics(results: dict):
    """Display key metrics in a row of cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total POIs",
            results.get('cleaned_poi_count', 0),
            f"Raw: {results.get('raw_poi_count', 0)}"
        )
    
    with col2:
        cluster_stats = results.get('cluster_stats', {})
        st.metric(
            "Clusters Found",
            cluster_stats.get('n_clusters', 0),
            f"Noise: {cluster_stats.get('n_noise', 0)}"
        )
    
    with col3:
        st.metric(
            "Hotspots",
            len(results.get('hotspots', [])),
            "High-density areas"
        )
    
    with col4:
        top_score = 0
        if results.get('top_locations') is not None and len(results['top_locations']) > 0:
            top_score = results['top_locations']['final_score'].max()
        st.metric(
            "Top Score",
            f"{top_score:.3f}",
            "Best location"
        )


def render_map_from_file(filepath: str, height: int = 500):
    """Render a saved HTML map file."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=height, scrolling=True)
    else:
        st.warning(f"Map file not found: {filepath}")


def create_live_map(map_type: str, center: tuple, cleaned_pois, 
                    clustered_pois, top_locations, cluster_stats,
                    business_type: str) -> folium.Map:
    """Create a live Folium map based on type."""
    viz = MapVisualizer(center[0], center[1], zoom_start=13)
    
    if map_type == "All POIs":
        return viz.create_all_pois_map(cleaned_pois)
    elif map_type == "Clusters":
        return viz.create_cluster_map(clustered_pois, cluster_stats)
    elif map_type == "Density Heatmap":
        return viz.create_heatmap(cleaned_pois)
    elif map_type == "Competition":
        return viz.create_competition_heatmap(cleaned_pois, business_type)
    elif map_type == "Recommendations":
        return viz.create_recommendations_map(cleaned_pois, top_locations, business_type)
    else:
        return viz.create_base_map()


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">📍 Business Location Recommender</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Geospatial Clustering & Location Recommendation System</p>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Configuration")
        
        city = st.text_input(
            "City Name",
            value="Bangalore",
            help="Enter the city name for analysis"
        )
        
        business_options = [
            'cafe', 'restaurant', 'shop', 'supermarket', 'pharmacy',
            'gym', 'hotel', 'bakery', 'fast_food', 'bank'
        ]
        business_type = st.selectbox(
            "Business Type",
            options=business_options,
            index=0,
            help="Select the type of business to analyze"
        )
        
        radius_km = st.slider(
            "Search Radius (km)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Radius around city center to search for POIs"
        )
        
        st.subheader("Scoring Weights")
        st.caption("Adjust weights to prioritize different factors")
        
        w_demand = st.slider("Demand Weight", 0.0, 1.0, 0.4, 0.05)
        w_competition = st.slider("Competition Weight", 0.0, 1.0, 0.3, 0.05)
        w_accessibility = st.slider("Accessibility Weight", 0.0, 1.0, 0.2, 0.05)
        w_infrastructure = st.slider("Infrastructure Weight", 0.0, 1.0, 0.1, 0.05)
        
        total = w_demand + w_competition + w_accessibility + w_infrastructure
        weights = {
            'demand': w_demand / total,
            'competition': w_competition / total,
            'accessibility': w_accessibility / total,
            'infrastructure': w_infrastructure / total
        }
        
        st.divider()
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if st.session_state.analysis_complete:
            st.success("Analysis Complete!")
    
    if run_analysis:
        st.session_state.analysis_complete = False
        
        with st.spinner("Running analysis..."):
            progress_bar = st.progress(0, "Initializing...")
            
            results = run_analysis_pipeline(
                city, business_type, radius_km, weights, progress_bar
            )
            
            if results:
                st.session_state.results = results
                st.session_state.analysis_complete = True
                st.rerun()
    
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        
        display_metrics(results)
        
        st.divider()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📍 POI Map",
            "🔵 Clusters",
            "🔥 Heatmaps",
            "⭐ Recommendations",
            "📊 Data & Reports"
        ])
        
        with tab1:
            st.subheader("All Points of Interest")
            st.caption("Interactive map showing all fetched POIs with marker clustering")
            render_map_from_file('maps/all_pois_map.html', height=600)
        
        with tab2:
            st.subheader("Cluster Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                render_map_from_file('maps/cluster_map.html', height=500)
            
            with col2:
                st.markdown("**Cluster Statistics**")
                cluster_stats = results.get('cluster_stats', {})
                st.write(f"Total Clusters: {cluster_stats.get('n_clusters', 0)}")
                st.write(f"Noise Points: {cluster_stats.get('n_noise', 0)}")
                
                if 'clusters' in cluster_stats:
                    st.markdown("**Top Clusters by Size:**")
                    clusters = cluster_stats['clusters']
                    sorted_clusters = sorted(
                        clusters.items(),
                        key=lambda x: x[1]['size'],
                        reverse=True
                    )[:5]
                    
                    for label, data in sorted_clusters:
                        st.write(f"Cluster {label}: {data['size']} POIs")
        
        with tab3:
            st.subheader("Density & Competition Heatmaps")
            
            heatmap_type = st.radio(
                "Select Heatmap",
                ["POI Density", "Competition"],
                horizontal=True
            )
            
            if heatmap_type == "POI Density":
                render_map_from_file('maps/heatmap.html', height=550)
            else:
                render_map_from_file('maps/competition_heatmap.html', height=550)
        
        with tab4:
            st.subheader("Top Recommended Locations")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                render_map_from_file('maps/recommendations_map.html', height=500)
            
            with col2:
                st.markdown("**Top 10 Locations**")
                
                top_locations = st.session_state.top_locations
                if top_locations is not None and len(top_locations) > 0:
                    for idx, row in top_locations.head(10).iterrows():
                        rank = int(row['rank'])
                        score = row['final_score']
                        
                        if rank <= 3:
                            st.markdown(f"🥇 **#{rank}** Score: {score:.3f}" if rank == 1 else 
                                       f"🥈 **#{rank}** Score: {score:.3f}" if rank == 2 else
                                       f"🥉 **#{rank}** Score: {score:.3f}")
                        else:
                            st.write(f"#{rank} Score: {score:.3f}")
                        
                        with st.expander(f"Details"):
                            st.write(f"Latitude: {row['latitude']:.5f}")
                            st.write(f"Longitude: {row['longitude']:.5f}")
                            st.write(f"Demand: {row['demand_score']:.3f}")
                            st.write(f"Competition: {row['competition_score']:.3f}")
                            st.write(f"Accessibility: {row['accessibility_score']:.3f}")
                            st.write(f"Infrastructure: {row['infrastructure_score']:.3f}")
        
        with tab5:
            st.subheader("Data Files & Analysis Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Download Data Files**")
                
                if os.path.exists('data/cleaned_data.csv'):
                    with open('data/cleaned_data.csv', 'r') as f:
                        st.download_button(
                            "📥 Download Cleaned POIs",
                            f.read(),
                            file_name="cleaned_pois.csv",
                            mime="text/csv"
                        )
                
                if os.path.exists('data/clusters.csv'):
                    with open('data/clusters.csv', 'r') as f:
                        st.download_button(
                            "📥 Download Clusters",
                            f.read(),
                            file_name="clusters.csv",
                            mime="text/csv"
                        )
                
                if os.path.exists('data/location_scores.csv'):
                    with open('data/location_scores.csv', 'r') as f:
                        st.download_button(
                            "📥 Download Location Scores",
                            f.read(),
                            file_name="location_scores.csv",
                            mime="text/csv"
                        )
            
            with col2:
                st.markdown("**Analysis Summary**")
                
                scoring_report = results.get('scoring_report', {})
                if scoring_report:
                    st.write(f"Candidates Analyzed: {scoring_report.get('total_candidates', 0)}")
                    st.write(f"Competitors Found: {scoring_report.get('total_competitors', 0)}")
                    st.write(f"Supporting POIs: {scoring_report.get('total_supporting_pois', 0)}")
                    st.write(f"Top Score: {scoring_report.get('top_score', 0):.3f}")
                
                st.markdown("**Weights Used:**")
                for key, value in weights.items():
                    st.write(f"  {key.title()}: {value:.2f}")
            
            st.divider()
            
            st.markdown("**Location Scores Table**")
            if os.path.exists('data/location_scores.csv'):
                scores_df = pd.read_csv('data/location_scores.csv')
                st.dataframe(
                    scores_df.head(20),
                    use_container_width=True,
                    hide_index=True
                )
    
    else:
        st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started.")
        
        st.markdown("### How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Data Collection**
            - Fetches POI data from OpenStreetMap
            - Includes shops, restaurants, banks, hospitals, schools, etc.
            - Configurable search radius
            """)
        
        with col2:
            st.markdown("""
            **2. Analysis**
            - DBSCAN clustering to find hotspots
            - Density and competition heatmaps
            - Multi-factor location scoring
            """)
        
        with col3:
            st.markdown("""
            **3. Recommendations**
            - Ranked list of best locations
            - Interactive map visualization
            - Downloadable reports
            """)
        
        st.markdown("---")
        st.markdown("*Powered by OSMnx, Folium, and Scikit-learn*")


if __name__ == "__main__":
    main()
