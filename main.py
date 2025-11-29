"""
Geospatial Clustering & Business Location Recommendation System
Main orchestration module for command-line execution

This module coordinates all system components to:
1. Fetch POI data from OpenStreetMap
2. Clean and prepare the data
3. Perform clustering analysis
4. Generate visualizations
5. Score and recommend optimal business locations
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from modules.fetch_data import DataFetcher
from modules.clean_data import DataCleaner
from modules.clustering import GeoClusterer
from modules.visualize import MapVisualizer
from modules.scoring import LocationScorer


def ensure_directories():
    """Create required directories if they don't exist."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('maps', exist_ok=True)


def run_analysis(city: str, business_type: str, radius_km: float = 5.0,
                 verbose: bool = True) -> dict:
    """
    Run the complete geospatial analysis pipeline.
    
    Args:
        city: Name of the city to analyze
        business_type: Type of business to recommend locations for
        radius_km: Search radius in kilometers
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary with analysis results and file paths
    """
    ensure_directories()
    results = {}
    
    if verbose:
        print("=" * 60)
        print("GEOSPATIAL CLUSTERING & BUSINESS LOCATION RECOMMENDATION")
        print("=" * 60)
        print(f"\nCity: {city}")
        print(f"Business Type: {business_type}")
        print(f"Search Radius: {radius_km} km")
        print("=" * 60)
    
    if verbose:
        print("\n[1/6] Fetching POI Data from OpenStreetMap...")
    
    fetcher = DataFetcher(city, radius_km)
    center = fetcher.get_city_center()
    
    if center is None:
        print(f"Error: Could not find city '{city}'")
        return results
    
    results['center'] = center
    if verbose:
        print(f"  City center: {center[0]:.5f}, {center[1]:.5f}")
    
    categories = [
        'shop', 'restaurant', 'cafe', 'supermarket', 'bank',
        'hospital', 'pharmacy', 'school', 'hotel', 'fuel',
        'parking', 'bus_station', business_type
    ]
    categories = list(set(categories))
    
    all_pois = fetcher.fetch_all_pois(categories)
    
    if len(all_pois) == 0:
        print("Error: No POIs found in the specified area")
        return results
    
    fetcher.save_to_csv(all_pois, 'raw_data.csv')
    results['raw_data_path'] = 'data/raw_data.csv'
    
    if verbose:
        print(f"  Total POIs fetched: {len(all_pois)}")
    
    if verbose:
        print("\n[2/6] Cleaning and Preparing Data...")
    
    cleaner = DataCleaner()
    cleaned_pois = cleaner.clean_geodataframe(all_pois)
    
    cleaning_report = cleaner.get_cleaning_report()
    if verbose:
        print(f"  Initial records: {cleaning_report['initial_count']}")
        print(f"  After cleaning: {cleaning_report['final_count']}")
        print(f"  Retention rate: {cleaning_report['retention_rate']:.1f}%")
    
    fetcher.save_to_csv(cleaned_pois, 'cleaned_data.csv')
    results['cleaned_data_path'] = 'data/cleaned_data.csv'
    results['poi_count'] = len(cleaned_pois)
    
    if verbose:
        print("\n[3/6] Performing Clustering Analysis...")
    
    coords = cleaner.prepare_for_clustering(cleaned_pois)
    
    clusterer = GeoClusterer()
    
    eps_km = min(radius_km / 10, 0.5)
    min_samples = max(3, int(len(coords) / 100))
    
    labels = clusterer.dbscan_clustering(coords, eps_km=eps_km, min_samples=min_samples)
    
    hotspots = clusterer.detect_hotspots(coords)
    sparse_regions = clusterer.detect_sparse_regions(
        coords, center[0], center[1]
    )
    
    cleaned_pois_with_clusters = clusterer.add_cluster_labels(cleaned_pois)
    clusterer.save_clusters(cleaned_pois_with_clusters)
    results['clusters_path'] = 'data/clusters.csv'
    
    cluster_summary = clusterer.get_cluster_summary()
    if verbose:
        print(cluster_summary)
    
    results['cluster_stats'] = clusterer.cluster_stats
    results['hotspots'] = hotspots
    results['sparse_regions'] = sparse_regions
    
    if verbose:
        print("\n[4/6] Generating Visualizations...")
    
    viz = MapVisualizer(center[0], center[1], zoom_start=13)
    
    all_pois_map = viz.create_all_pois_map(cleaned_pois)
    viz.save_map(all_pois_map, 'all_pois_map.html')
    results['all_pois_map'] = 'maps/all_pois_map.html'
    if verbose:
        print("  Created: all_pois_map.html")
    
    cluster_map = viz.create_cluster_map(
        cleaned_pois_with_clusters,
        clusterer.cluster_stats
    )
    viz.save_map(cluster_map, 'cluster_map.html')
    results['cluster_map'] = 'maps/cluster_map.html'
    if verbose:
        print("  Created: cluster_map.html")
    
    heatmap = viz.create_heatmap(cleaned_pois)
    viz.save_map(heatmap, 'heatmap.html')
    results['heatmap'] = 'maps/heatmap.html'
    if verbose:
        print("  Created: heatmap.html")
    
    competition_map = viz.create_competition_heatmap(cleaned_pois, business_type)
    viz.save_map(competition_map, 'competition_heatmap.html')
    results['competition_heatmap'] = 'maps/competition_heatmap.html'
    if verbose:
        print("  Created: competition_heatmap.html")
    
    if verbose:
        print("\n[5/6] Scoring Potential Locations...")
    
    scorer = LocationScorer()
    
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
    results['scores_path'] = 'data/location_scores.csv'
    
    top_locations = scorer.get_top_locations(10)
    results['top_locations'] = top_locations
    
    if verbose:
        print(scorer.print_report())
    
    if verbose:
        print("\n[6/6] Creating Recommendations Map...")
    
    recommendations_map = viz.create_recommendations_map(
        cleaned_pois, top_locations, business_type
    )
    viz.save_map(recommendations_map, 'recommendations_map.html')
    results['recommendations_map'] = 'maps/recommendations_map.html'
    if verbose:
        print("  Created: recommendations_map.html")
    
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nGenerated Files:")
        print("  Data:")
        print("    - data/raw_data.csv")
        print("    - data/cleaned_data.csv")
        print("    - data/clusters.csv")
        print("    - data/location_scores.csv")
        print("  Maps:")
        print("    - maps/all_pois_map.html")
        print("    - maps/cluster_map.html")
        print("    - maps/heatmap.html")
        print("    - maps/competition_heatmap.html")
        print("    - maps/recommendations_map.html")
        print("\nTop 3 Recommended Locations:")
        for idx, row in top_locations.head(3).iterrows():
            print(f"  {int(row['rank'])}. ({row['latitude']:.5f}, {row['longitude']:.5f}) "
                  f"- Score: {row['final_score']:.3f}")
        print("=" * 60)
    
    return results


def interactive_mode():
    """Run in interactive mode, prompting for user input."""
    print("\n" + "=" * 60)
    print("GEOSPATIAL BUSINESS LOCATION RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    city = input("\nEnter city name: ").strip()
    if not city:
        city = "Bangalore"
        print(f"Using default: {city}")
    
    business_type = input("Enter business type (e.g., coffee_shop, restaurant): ").strip()
    if not business_type:
        business_type = "cafe"
        print(f"Using default: {business_type}")
    
    radius_input = input("Enter search radius in km (default 5): ").strip()
    try:
        radius_km = float(radius_input) if radius_input else 5.0
    except ValueError:
        radius_km = 5.0
        print(f"Invalid input, using default: {radius_km} km")
    
    results = run_analysis(city, business_type, radius_km)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Geospatial Clustering & Business Location Recommendation System'
    )
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--business', '-b', type=str, help='Business type')
    parser.add_argument('--radius', '-r', type=float, default=5.0,
                        help='Search radius in km (default: 5)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.city and not args.business):
        results = interactive_mode()
    elif args.city and args.business:
        results = run_analysis(args.city, args.business, args.radius)
    else:
        print("Please provide both --city and --business, or use --interactive mode")
        parser.print_help()
        return
    
    if not results:
        print("Analysis failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
