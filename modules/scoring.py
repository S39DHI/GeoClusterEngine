"""
Location Scoring Module
Computes suitability scores for potential business locations
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler


class LocationScorer:
    """
    Implements the location scoring model for business site selection.
    
    Final Score = w1 * demand_score + w2 * low_competition_score + 
                  w3 * accessibility_score + w4 * infrastructure_score
    
    Default weights:
    - Demand: 0.4
    - Competition: 0.3
    - Accessibility: 0.2
    - Infrastructure: 0.1
    """
    
    DEFAULT_WEIGHTS = {
        'demand': 0.4,
        'competition': 0.3,
        'accessibility': 0.2,
        'infrastructure': 0.1
    }
    
    INFRASTRUCTURE_CATEGORIES = ['hospital', 'bank', 'school', 'pharmacy']
    TRANSPORT_CATEGORIES = ['bus_station', 'parking', 'fuel']
    
    def __init__(self, weights: dict = None):
        """
        Initialize the LocationScorer.
        
        Args:
            weights: Custom weights for scoring components
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.scaler = MinMaxScaler()
        self.scored_locations = None
        self.analysis_report = {}
    
    def set_weights(self, demand: float = None, competition: float = None,
                    accessibility: float = None, infrastructure: float = None):
        """
        Set custom weights for scoring components.
        
        Args:
            demand: Weight for demand score (0-1)
            competition: Weight for competition score (0-1)
            accessibility: Weight for accessibility score (0-1)
            infrastructure: Weight for infrastructure score (0-1)
        """
        if demand is not None:
            self.weights['demand'] = demand
        if competition is not None:
            self.weights['competition'] = competition
        if accessibility is not None:
            self.weights['accessibility'] = accessibility
        if infrastructure is not None:
            self.weights['infrastructure'] = infrastructure
        
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def compute_demand_score(self, candidate_coords: np.ndarray,
                            supporting_pois: gpd.GeoDataFrame,
                            radius_km: float = 1.0) -> np.ndarray:
        """
        Compute demand score based on supporting POIs density.
        
        Args:
            candidate_coords: Array of candidate location coordinates (lat, lon)
            supporting_pois: GeoDataFrame of supporting POIs
            radius_km: Search radius in km
            
        Returns:
            Array of demand scores
        """
        if len(supporting_pois) == 0 or len(candidate_coords) == 0:
            return np.zeros(len(candidate_coords))
        
        poi_coords = supporting_pois[['latitude', 'longitude']].values
        poi_coords = poi_coords[~np.isnan(poi_coords).any(axis=1)]
        
        if len(poi_coords) == 0:
            return np.zeros(len(candidate_coords))
        
        tree = cKDTree(poi_coords)
        
        radius_deg = radius_km / 111.0
        
        counts = []
        for coord in candidate_coords:
            nearby = tree.query_ball_point(coord, radius_deg)
            counts.append(len(nearby))
        
        counts = np.array(counts, dtype=float)
        
        if counts.max() > 0:
            counts = counts / counts.max()
        
        return counts
    
    def compute_competition_score(self, candidate_coords: np.ndarray,
                                  competitors: gpd.GeoDataFrame,
                                  radius_km: float = 0.5) -> np.ndarray:
        """
        Compute competition score (inverse of competitor density).
        Lower competition = higher score.
        
        Args:
            candidate_coords: Array of candidate location coordinates (lat, lon)
            competitors: GeoDataFrame of competitor locations
            radius_km: Search radius in km
            
        Returns:
            Array of competition scores (higher = less competition)
        """
        if len(competitors) == 0 or len(candidate_coords) == 0:
            return np.ones(len(candidate_coords))
        
        comp_coords = competitors[['latitude', 'longitude']].values
        comp_coords = comp_coords[~np.isnan(comp_coords).any(axis=1)]
        
        if len(comp_coords) == 0:
            return np.ones(len(candidate_coords))
        
        tree = cKDTree(comp_coords)
        
        radius_deg = radius_km / 111.0
        
        counts = []
        for coord in candidate_coords:
            nearby = tree.query_ball_point(coord, radius_deg)
            counts.append(len(nearby))
        
        counts = np.array(counts, dtype=float)
        
        max_count = counts.max()
        if max_count > 0:
            scores = 1 - (counts / (max_count + 1))
        else:
            scores = np.ones(len(candidate_coords))
        
        return scores
    
    def compute_accessibility_score(self, candidate_coords: np.ndarray,
                                   transport_pois: gpd.GeoDataFrame = None,
                                   road_network: gpd.GeoDataFrame = None,
                                   radius_km: float = 0.5) -> np.ndarray:
        """
        Compute accessibility score based on transport and roads.
        
        Args:
            candidate_coords: Array of candidate location coordinates (lat, lon)
            transport_pois: GeoDataFrame of transport-related POIs
            road_network: GeoDataFrame of road network
            radius_km: Search radius in km
            
        Returns:
            Array of accessibility scores
        """
        scores = np.zeros(len(candidate_coords))
        
        if transport_pois is not None and len(transport_pois) > 0:
            transport_coords = transport_pois[['latitude', 'longitude']].values
            transport_coords = transport_coords[~np.isnan(transport_coords).any(axis=1)]
            
            if len(transport_coords) > 0:
                tree = cKDTree(transport_coords)
                radius_deg = radius_km / 111.0
                
                for i, coord in enumerate(candidate_coords):
                    nearby = tree.query_ball_point(coord, radius_deg)
                    scores[i] = len(nearby)
        
        if road_network is not None and len(road_network) > 0:
            try:
                if 'geometry' in road_network.columns:
                    road_coords = []
                    for geom in road_network.geometry:
                        if geom and not geom.is_empty:
                            centroid = geom.centroid
                            road_coords.append([centroid.y, centroid.x])
                    
                    if road_coords:
                        road_coords = np.array(road_coords)
                        tree = cKDTree(road_coords)
                        radius_deg = radius_km / 111.0
                        
                        for i, coord in enumerate(candidate_coords):
                            nearby = tree.query_ball_point(coord, radius_deg)
                            scores[i] += len(nearby) * 0.5
            except Exception as e:
                pass
        
        if scores.max() > 0:
            scores = scores / scores.max()
        else:
            scores = np.ones(len(candidate_coords)) * 0.5
        
        return scores
    
    def compute_infrastructure_score(self, candidate_coords: np.ndarray,
                                    infrastructure_pois: gpd.GeoDataFrame,
                                    radius_km: float = 1.0) -> np.ndarray:
        """
        Compute infrastructure score based on nearby essential services.
        
        Args:
            candidate_coords: Array of candidate location coordinates (lat, lon)
            infrastructure_pois: GeoDataFrame with hospitals, banks, schools
            radius_km: Search radius in km
            
        Returns:
            Array of infrastructure scores
        """
        if len(infrastructure_pois) == 0 or len(candidate_coords) == 0:
            return np.zeros(len(candidate_coords))
        
        infra_pois = infrastructure_pois[
            infrastructure_pois['category'].str.lower().isin(
                [c.lower() for c in self.INFRASTRUCTURE_CATEGORIES]
            )
        ]
        
        if len(infra_pois) == 0:
            return np.zeros(len(candidate_coords))
        
        infra_coords = infra_pois[['latitude', 'longitude']].values
        infra_coords = infra_coords[~np.isnan(infra_coords).any(axis=1)]
        
        if len(infra_coords) == 0:
            return np.zeros(len(candidate_coords))
        
        tree = cKDTree(infra_coords)
        radius_deg = radius_km / 111.0
        
        scores = []
        for coord in candidate_coords:
            nearby = tree.query_ball_point(coord, radius_deg)
            scores.append(len(nearby))
        
        scores = np.array(scores, dtype=float)
        
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def generate_candidate_locations(self, center_lat: float, center_lon: float,
                                    radius_km: float = 5.0,
                                    grid_size: int = 20) -> np.ndarray:
        """
        Generate a grid of candidate locations.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Radius in km
            grid_size: Number of grid points per dimension
            
        Returns:
            Array of candidate coordinates
        """
        lat_range = radius_km / 111.0
        lon_range = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        lats = np.linspace(center_lat - lat_range, center_lat + lat_range, grid_size)
        lons = np.linspace(center_lon - lon_range, center_lon + lon_range, grid_size)
        
        candidates = []
        for lat in lats:
            for lon in lons:
                dist = np.sqrt(
                    ((lat - center_lat) * 111.0) ** 2 +
                    ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))) ** 2
                )
                if dist <= radius_km:
                    candidates.append([lat, lon])
        
        return np.array(candidates)
    
    def score_locations(self, candidate_coords: np.ndarray,
                       all_pois: gpd.GeoDataFrame,
                       business_type: str,
                       supporting_categories: list = None) -> pd.DataFrame:
        """
        Score all candidate locations.
        
        Args:
            candidate_coords: Array of candidate coordinates
            all_pois: GeoDataFrame with all POIs
            business_type: Type of business to analyze
            supporting_categories: List of supporting POI categories
            
        Returns:
            DataFrame with scored locations
        """
        if supporting_categories is None:
            supporting_categories = ['shop', 'restaurant', 'cafe', 'supermarket']
        
        competitors = all_pois[
            all_pois['category'].str.lower() == business_type.lower()
        ]
        
        supporting_mask = all_pois['category'].str.lower().isin(
            [c.lower() for c in supporting_categories]
        )
        supporting_pois = all_pois[supporting_mask]
        
        transport_mask = all_pois['category'].str.lower().isin(
            [c.lower() for c in self.TRANSPORT_CATEGORIES]
        )
        transport_pois = all_pois[transport_mask]
        
        demand_scores = self.compute_demand_score(
            candidate_coords, supporting_pois
        )
        competition_scores = self.compute_competition_score(
            candidate_coords, competitors
        )
        accessibility_scores = self.compute_accessibility_score(
            candidate_coords, transport_pois
        )
        infrastructure_scores = self.compute_infrastructure_score(
            candidate_coords, all_pois
        )
        
        final_scores = (
            self.weights['demand'] * demand_scores +
            self.weights['competition'] * competition_scores +
            self.weights['accessibility'] * accessibility_scores +
            self.weights['infrastructure'] * infrastructure_scores
        )
        
        results = pd.DataFrame({
            'latitude': candidate_coords[:, 0],
            'longitude': candidate_coords[:, 1],
            'demand_score': demand_scores,
            'competition_score': competition_scores,
            'accessibility_score': accessibility_scores,
            'infrastructure_score': infrastructure_scores,
            'final_score': final_scores
        })
        
        results = results.sort_values('final_score', ascending=False)
        results['rank'] = range(1, len(results) + 1)
        
        self.scored_locations = results
        
        self.analysis_report = {
            'total_candidates': len(candidate_coords),
            'total_competitors': len(competitors),
            'total_supporting_pois': len(supporting_pois),
            'avg_demand_score': float(demand_scores.mean()),
            'avg_competition_score': float(competition_scores.mean()),
            'top_score': float(results['final_score'].max()),
            'weights_used': self.weights.copy()
        }
        
        return results
    
    def get_top_locations(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N recommended locations.
        
        Args:
            n: Number of locations to return
            
        Returns:
            DataFrame with top locations
        """
        if self.scored_locations is None:
            raise ValueError("Run score_locations first")
            
        return self.scored_locations.head(n)
    
    def save_scores(self, filepath: str = 'data/location_scores.csv') -> str:
        """
        Save scored locations to CSV.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to saved file
        """
        if self.scored_locations is None:
            raise ValueError("Run score_locations first")
            
        self.scored_locations.to_csv(filepath, index=False)
        print(f"Saved location scores to {filepath}")
        return filepath
    
    def get_analysis_report(self) -> dict:
        """
        Get the analysis report.
        
        Returns:
            Dictionary with analysis statistics
        """
        return self.analysis_report
    
    def print_report(self) -> str:
        """
        Generate a printable analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.analysis_report:
            return "No analysis performed yet"
            
        report = []
        report.append("=" * 50)
        report.append("LOCATION SCORING ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Candidate locations analyzed: {self.analysis_report.get('total_candidates', 0)}")
        report.append(f"Competitors found: {self.analysis_report.get('total_competitors', 0)}")
        report.append(f"Supporting POIs: {self.analysis_report.get('total_supporting_pois', 0)}")
        report.append("")
        report.append("Scoring Weights Used:")
        for key, value in self.analysis_report.get('weights_used', {}).items():
            report.append(f"  - {key}: {value:.2f}")
        report.append("")
        report.append("Score Statistics:")
        report.append(f"  - Top Score: {self.analysis_report.get('top_score', 0):.3f}")
        report.append(f"  - Avg Demand Score: {self.analysis_report.get('avg_demand_score', 0):.3f}")
        report.append(f"  - Avg Competition Score: {self.analysis_report.get('avg_competition_score', 0):.3f}")
        report.append("")
        
        if self.scored_locations is not None:
            report.append("TOP 5 RECOMMENDED LOCATIONS:")
            report.append("-" * 40)
            for idx, row in self.scored_locations.head(5).iterrows():
                report.append(
                    f"  #{int(row['rank'])}. ({row['latitude']:.5f}, {row['longitude']:.5f}) "
                    f"- Score: {row['final_score']:.3f}"
                )
        
        report.append("=" * 50)
        
        return "\n".join(report)
