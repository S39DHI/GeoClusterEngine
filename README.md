# Geospatial Clustering & Business Location Recommendation System

A comprehensive Python-based system that collects, cleans, visualizes, and clusters geospatial POI (Points of Interest) data to find the best location for a new business. The system computes demand, competition, accessibility, and infrastructure scores to recommend ideal locations.

## Features

- **Data Collection**: Fetches POI data from OpenStreetMap using OSMnx
- **Data Cleaning**: Removes duplicates, validates coordinates, normalizes categories
- **Clustering Analysis**: DBSCAN with haversine distance metric, K-Means comparison
- **Interactive Visualizations**: Folium maps with marker clustering, heatmaps
- **Location Scoring**: Multi-factor scoring model for site selection
- **Streamlit Dashboard**: Interactive web interface for analysis

## Project Structure

```
/project
    /data
        raw_data.csv           # Original fetched POI data
        cleaned_data.csv       # Processed and cleaned data
        clusters.csv           # POIs with cluster labels
        location_scores.csv    # Scored candidate locations
    /maps
        all_pois_map.html      # Interactive map of all POIs
        cluster_map.html       # Cluster visualization map
        heatmap.html           # POI density heatmap
        competition_heatmap.html # Competition analysis map
        recommendations_map.html # Top recommended locations
    /modules
        __init__.py            # Package initialization
        fetch_data.py          # OSM data fetching module
        clean_data.py          # Data cleaning module
        clustering.py          # Clustering algorithms
        visualize.py           # Map visualization module
        scoring.py             # Location scoring module
    main.py                    # Command-line orchestration
    app.py                     # Streamlit dashboard
    README.md                  # This file
```

## Installation

The project uses the following Python packages:

```bash
pip install osmnx geopandas folium scikit-learn pandas numpy shapely streamlit streamlit-folium
```

## How to Run

### Option 1: Streamlit Dashboard (Recommended)

```bash
streamlit run app.py --server.port 5000
```

Then open your browser to the provided URL to access the interactive dashboard.

### Option 2: Command Line Interface

```bash
# Interactive mode
python main.py --interactive

# Direct execution
python main.py --city "Bangalore" --business "cafe" --radius 5
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--city` | `-c` | City name to analyze |
| `--business` | `-b` | Business type (cafe, restaurant, shop, etc.) |
| `--radius` | `-r` | Search radius in kilometers (default: 5) |
| `--interactive` | `-i` | Run in interactive mode |

## Algorithms Used

### 1. DBSCAN Clustering

- **Purpose**: Identifies dense clusters of POIs and noise points
- **Distance Metric**: Haversine distance (accounts for Earth's curvature)
- **Parameters**:
  - `eps`: Maximum distance between points (in radians)
  - `min_samples`: Minimum points to form a cluster

### 2. K-Means Clustering

- **Purpose**: Alternative clustering for comparison
- **Features**: Finds K cluster centers using iterative optimization
- **Used for**: Optimal cluster count estimation via silhouette score

### 3. Kernel Density Estimation (KDE)

- **Purpose**: Creates smooth density surfaces for heatmaps
- **Implementation**: Gaussian kernel over coordinate space
- **Output**: Interactive Folium heatmap layers

### 4. Location Scoring Model

```
Final Score = w1 * demand_score + 
              w2 * competition_score + 
              w3 * accessibility_score + 
              w4 * infrastructure_score
```

**Default Weights:**
- Demand: 0.4 (40%)
- Competition: 0.3 (30%)
- Accessibility: 0.2 (20%)
- Infrastructure: 0.1 (10%)

**Score Components:**

| Score | Description | Calculation |
|-------|-------------|-------------|
| Demand | Supporting POIs density | Count of shops, restaurants nearby |
| Competition | Inverse competitor density | Lower competition = higher score |
| Accessibility | Transport access | Bus stops, parking, major roads |
| Infrastructure | Essential services | Hospitals, banks, schools nearby |

## POI Categories Supported

- **Retail**: shop, supermarket, convenience, mall
- **Food & Beverage**: restaurant, cafe, fast_food, bakery
- **Finance**: bank, atm
- **Healthcare**: hospital, pharmacy, clinic
- **Education**: school, university, college
- **Hospitality**: hotel, hostel
- **Transport**: bus_station, parking, fuel
- **Leisure**: gym, cinema

## Output Files

### CSV Files (in `/data/`)

1. **raw_data.csv**: Original fetched POI data
2. **cleaned_data.csv**: Processed data with validated coordinates
3. **clusters.csv**: POIs with cluster assignments and hotspot flags
4. **location_scores.csv**: Ranked candidate locations with scores

### HTML Maps (in `/maps/`)

1. **all_pois_map.html**: All POIs with marker clustering
2. **cluster_map.html**: Color-coded cluster visualization
3. **heatmap.html**: POI density heatmap
4. **competition_heatmap.html**: Competition concentration map
5. **recommendations_map.html**: Top 10 recommended locations

## Dashboard Features

The Streamlit dashboard provides:

- **Configuration Panel**: City, business type, radius, and weight adjustments
- **POI Map Tab**: Interactive map of all fetched points of interest
- **Clusters Tab**: Cluster visualization with statistics
- **Heatmaps Tab**: Toggle between density and competition heatmaps
- **Recommendations Tab**: Top 10 locations with detailed scores
- **Data & Reports Tab**: Download CSV files and view analysis summary

## Example Usage

### Dashboard Workflow

1. Enter city name (e.g., "Bangalore", "Mumbai", "New York")
2. Select business type from dropdown
3. Adjust search radius (1-20 km)
4. Optionally modify scoring weights
5. Click "Run Analysis"
6. Explore maps and recommendations in different tabs
7. Download data files for further analysis

### Console Output Example

```
============================================================
GEOSPATIAL CLUSTERING & BUSINESS LOCATION RECOMMENDATION
============================================================

City: Bangalore
Business Type: cafe
Search Radius: 5 km

[1/6] Fetching POI Data from OpenStreetMap...
  City center: 12.97194, 77.59369
  Fetching shop... Found 523 POIs
  Fetching restaurant... Found 312 POIs
  ...

[2/6] Cleaning and Preparing Data...
  Initial records: 1842
  After cleaning: 1756
  Retention rate: 95.3%

[3/6] Performing Clustering Analysis...
  Total clusters found: 15
  Noise points: 127
  Hotspots identified: 4

[4/6] Generating Visualizations...
  Created: all_pois_map.html
  Created: cluster_map.html
  ...

[5/6] Scoring Potential Locations...
  Candidates analyzed: 177
  Top Score: 0.847

[6/6] Creating Recommendations Map...

ANALYSIS COMPLETE!
============================================================
```

## Future Improvements

1. **Real-time Data Integration**
   - Google Places API for ratings and reviews
   - Population density data from census APIs
   - Real-time foot traffic data

2. **Advanced Analytics**
   - Time-series analysis for seasonal patterns
   - Competitor performance prediction
   - Market saturation modeling

3. **Enhanced Visualization**
   - 3D terrain visualization
   - Catchment area analysis
   - Isochrone maps for travel time

4. **Machine Learning Enhancements**
   - HDBSCAN for variable-density clustering
   - Neural network-based scoring
   - Demand prediction models

5. **Export Options**
   - PDF report generation
   - GeoJSON export for GIS software
   - API endpoint for integration

## Technical Notes

- The system uses the WGS84 (EPSG:4326) coordinate reference system
- Haversine distance calculations account for Earth's curvature
- KD-Tree data structure used for efficient nearest-neighbor queries
- Memory-efficient streaming for large datasets

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| osmnx | 2.0+ | OpenStreetMap data fetching |
| geopandas | 0.14+ | Geospatial data handling |
| folium | 0.15+ | Interactive map creation |
| scikit-learn | 1.3+ | Clustering algorithms |
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| shapely | 2.0+ | Geometric operations |
| streamlit | 1.28+ | Web dashboard |
| streamlit-folium | 0.15+ | Folium-Streamlit integration |

## License

This project is open source and available for educational and commercial use.

## Acknowledgments

- OpenStreetMap contributors for POI data
- OSMnx developers for the excellent Python interface
- Folium team for interactive mapping capabilities
