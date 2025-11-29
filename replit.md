# Geospatial Clustering & Business Location Recommendation System

## Overview

A comprehensive Python-based geospatial analysis system that helps find the best locations for new businesses by analyzing Points of Interest (POI) data from OpenStreetMap. The system uses clustering algorithms, heatmap visualizations, and multi-factor scoring to recommend optimal business locations.

## Project Architecture

```
/project
├── app.py                 # Streamlit web dashboard (main entry point)
├── main.py                # Command-line orchestration script
├── modules/               # Core analysis modules
│   ├── __init__.py       # Package initialization
│   ├── fetch_data.py     # OSM data fetching with OSMnx
│   ├── clean_data.py     # Data cleaning and validation
│   ├── clustering.py     # DBSCAN and K-Means clustering
│   ├── visualize.py      # Folium map generation
│   └── scoring.py        # Location scoring model
├── data/                  # Generated CSV outputs
│   ├── raw_data.csv      # Original POI data
│   ├── cleaned_data.csv  # Processed data
│   ├── clusters.csv      # Clustered POIs
│   └── location_scores.csv # Scored locations
├── maps/                  # Generated HTML maps
│   ├── all_pois_map.html
│   ├── cluster_map.html
│   ├── heatmap.html
│   ├── competition_heatmap.html
│   └── recommendations_map.html
└── README.md              # Full documentation
```

## Key Components

### 1. Data Fetching (fetch_data.py)
- Uses OSMnx to query OpenStreetMap
- Fetches POIs: shops, restaurants, banks, hospitals, schools, etc.
- Exports to CSV and GeoJSON formats

### 2. Data Cleaning (clean_data.py)
- Removes duplicates and invalid coordinates
- Normalizes category labels
- Converts to GeoDataFrame format

### 3. Clustering (clustering.py)
- DBSCAN with haversine distance metric
- K-Means for comparison
- Hotspot and sparse region detection

### 4. Visualization (visualize.py)
- Folium maps with marker clustering
- Density and competition heatmaps
- KDE-based visualizations

### 5. Scoring (scoring.py)
- Multi-factor location scoring model
- Weights: demand (40%), competition (30%), accessibility (20%), infrastructure (10%)
- Returns ranked list of recommended locations

## Running the Application

### Streamlit Dashboard (Default)
```bash
streamlit run app.py --server.port 5000
```

### Command Line
```bash
python main.py --city "Bangalore" --business "cafe" --radius 5
python main.py --interactive
```

## Technology Stack

- **Data**: OSMnx, GeoPandas, Pandas, NumPy
- **Clustering**: Scikit-learn (DBSCAN, K-Means)
- **Visualization**: Folium, Streamlit-Folium
- **Web Framework**: Streamlit

## Recent Changes

- Initial project setup with complete module structure
- Implemented all 5 core modules
- Created interactive Streamlit dashboard
- Added comprehensive documentation

## User Preferences

- Default city: Bangalore
- Default business type: cafe
- Default search radius: 5 km
