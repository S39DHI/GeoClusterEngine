"""
Business Suggestion Module
Recommends suitable business types for a given location
"""

import pandas as pd
import numpy as np
import random
from modules.clean_data import DataCleaner
from modules.scoring import LocationScorer


class BusinessSuggester:
    """
    Analyzes a specific location and suggests viable business types.
    """
    
    BUSINESS_TYPES = [
        'cafe', 'restaurant', 'bakery', 'fast_food', 'shop',
        'supermarket', 'pharmacy', 'bank', 'gym', 'hotel'
    ]
    
    def __init__(self, cleaned_pois: pd.DataFrame, center_lat: float, center_lon: float):
        """
        Initialize the BusinessSuggester.
        
        Args:
            cleaned_pois: GeoDataFrame of all POIs in the area
            center_lat: Latitude of city center
            center_lon: Longitude of city center
        """
        self.cleaned_pois = cleaned_pois
        self.center_lat = center_lat
        self.center_lon = center_lon
    
    def suggest_businesses_at_location(self, lat: float, lon: float, 
                                       radius_km: float = 5.0) -> pd.DataFrame:
        """
        Suggest viable business types for a given location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            radius_km: Search radius in km
            
        Returns:
            DataFrame with business suggestions ranked by viability score
        """
        suggestions = []
        
        # Use equal weights for comprehensive analysis
        weights = {
            'demand': 0.25,
            'competition': 0.25,
            'accessibility': 0.25,
            'infrastructure': 0.25
        }
        
        for business_type in self.BUSINESS_TYPES:
            try:
                # Get local POIs around the target location
                local_pois = self._get_nearby_pois(lat, lon, radius_km)
                
                # Count businesses of this type (competition)
                business_pois = local_pois[
                    local_pois['category'].str.contains(business_type, case=False, na=False)
                ] if 'category' in local_pois.columns else pd.DataFrame()
                
                competition_density = len(business_pois)
                
                # Calculate demand score based on nearby supporting POIs
                demand_score = self._calculate_demand_score(local_pois, business_type)
                
                # Calculate competition score (inverse - fewer competitors = higher score)
                competition_score = max(0, 1.0 - (competition_density / max(1, len(local_pois))))
                
                # Calculate accessibility score based on transport/infrastructure
                accessibility_score = self._calculate_accessibility_score(local_pois)
                
                # Calculate infrastructure score
                infrastructure_score = self._calculate_infrastructure_score(local_pois)
                
                # Compute overall viability score
                viability_score = (
                    weights['demand'] * demand_score +
                    weights['competition'] * competition_score +
                    weights['accessibility'] * accessibility_score +
                    weights['infrastructure'] * infrastructure_score
                )
                
                suggestions.append({
                    'Business Type': business_type.title(),
                    'Viability Score': float(viability_score),
                    'Demand': float(demand_score),
                    'Competition': float(competition_score),
                    'Accessibility': float(accessibility_score),
                    'Infrastructure': float(infrastructure_score),
                    'Local Competitors': competition_density,
                    'Category': business_type
                })
            except Exception as e:
                # If scoring fails for a business type, try simplified scoring
                try:
                    local_pois = self._get_nearby_pois(lat, lon, radius_km)
                    business_pois = local_pois[
                        local_pois['category'].str.contains(business_type, case=False, na=False)
                    ] if 'category' in local_pois.columns else pd.DataFrame()
                    
                    competition_density = len(business_pois)
                    poi_density = len(local_pois) / max(1, radius_km)
                    
                    # Simple scoring based on POI density and competition
                    demand_score = min(1.0, poi_density / 50.0)
                    competition_score = max(0, 1.0 - (competition_density / max(1, len(local_pois))))
                    
                    viability_score = 0.5 * demand_score + 0.5 * competition_score
                    
                    suggestions.append({
                        'Business Type': business_type.title(),
                        'Viability Score': float(viability_score),
                        'Demand': float(demand_score),
                        'Competition': float(competition_score),
                        'Accessibility': 0.5,
                        'Infrastructure': 0.5,
                        'Local Competitors': competition_density,
                        'Category': business_type
                    })
                except:
                    # Final fallback
                    suggestions.append({
                        'Business Type': business_type.title(),
                        'Viability Score': 0.0,
                        'Demand': 0.0,
                        'Competition': 0.0,
                        'Accessibility': 0.0,
                        'Infrastructure': 0.0,
                        'Local Competitors': 0,
                        'Category': business_type
                    })
        
        # Create DataFrame and sort by viability
        results = pd.DataFrame(suggestions)
        results = results.sort_values('Viability Score', ascending=False)
        results['Rank'] = range(1, len(results) + 1)
        
        return results[['Rank', 'Business Type', 'Viability Score', 'Demand', 
                        'Competition', 'Accessibility', 'Infrastructure', 
                        'Local Competitors']]
    
    def _get_nearby_pois(self, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
        """
        Get POIs within a specific radius of a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Radius in kilometers
            
        Returns:
            Filtered GeoDataFrame
        """
        from scipy.spatial.distance import haversine
        
        if len(self.cleaned_pois) == 0:
            return pd.DataFrame()
        
        distances = self.cleaned_pois.apply(
            lambda row: haversine(
                (lat, lon),
                (row['latitude'], row['longitude'])
            ),
            axis=1
        )
        
        return self.cleaned_pois[distances <= radius_km].copy()
    
    def _calculate_demand_score(self, local_pois: pd.DataFrame, business_type: str) -> float:
        """
        Calculate demand score based on supporting POIs.
        
        Args:
            local_pois: POIs in the local area
            business_type: Type of business
            
        Returns:
            Demand score between 0-1
        """
        if len(local_pois) == 0:
            return 0.0
        
        from modules.fetch_data import DataFetcher
        fetcher = DataFetcher("", 1.0)
        supporting = fetcher.get_supporting_pois(business_type)
        
        # Count supporting POIs
        supporting_count = 0
        for cat in supporting:
            count = len(local_pois[
                local_pois['category'].str.contains(cat, case=False, na=False)
            ]) if 'category' in local_pois.columns else 0
            supporting_count += count
        
        # Normalize to 0-1 scale
        demand_score = min(1.0, supporting_count / max(1, len(supporting)))
        return demand_score
    
    def _calculate_accessibility_score(self, local_pois: pd.DataFrame) -> float:
        """
        Calculate accessibility score based on transport/parking POIs.
        
        Args:
            local_pois: POIs in the local area
            
        Returns:
            Accessibility score between 0-1
        """
        if len(local_pois) == 0:
            return 0.0
        
        transport_keywords = ['bus', 'station', 'parking', 'fuel', 'transport']
        transport_pois = 0
        
        if 'category' in local_pois.columns:
            for keyword in transport_keywords:
                transport_pois += len(local_pois[
                    local_pois['category'].str.contains(keyword, case=False, na=False)
                ])
        
        accessibility_score = min(1.0, transport_pois / max(1, len(local_pois)))
        return accessibility_score
    
    def _calculate_infrastructure_score(self, local_pois: pd.DataFrame) -> float:
        """
        Calculate infrastructure score based on essential services.
        
        Args:
            local_pois: POIs in the local area
            
        Returns:
            Infrastructure score between 0-1
        """
        if len(local_pois) == 0:
            return 0.0
        
        infrastructure_keywords = ['hospital', 'school', 'bank', 'pharmacy']
        infra_pois = 0
        
        if 'category' in local_pois.columns:
            for keyword in infrastructure_keywords:
                infra_pois += len(local_pois[
                    local_pois['category'].str.contains(keyword, case=False, na=False)
                ])
        
        infrastructure_score = min(1.0, infra_pois / max(1, len(local_pois)))
        return infrastructure_score

    
    def get_recommendation_summary(self, suggestions: pd.DataFrame) -> str:
        """
        Get a text summary of business suggestions.
        
        Args:
            suggestions: DataFrame from suggest_businesses_at_location
            
        Returns:
            Summary string
        """
        if len(suggestions) == 0:
            return "No business suggestions available."
        
        summary = []
        summary.append("=" * 60)
        summary.append("BUSINESS VIABILITY ANALYSIS FOR LOCATION")
        summary.append("=" * 60)
        summary.append("")
        
        # Top 3 recommendations
        summary.append("🏆 TOP RECOMMENDED BUSINESSES:")
        for idx, row in suggestions.head(3).iterrows():
            medal = "🥇" if row['Rank'] == 1 else "🥈" if row['Rank'] == 2 else "🥉"
            summary.append(
                f"{medal} #{row['Rank']}. {row['Business Type']}: "
                f"Score {row['Viability Score']:.3f}"
            )
        
        summary.append("")
        summary.append("📊 ANALYSIS:")
        
        top_business = suggestions.iloc[0]
        summary.append(f"Best option: {top_business['Business Type']}")
        summary.append(f"  - Demand: {top_business['Demand']:.3f}/1.0")
        summary.append(f"  - Competition: {top_business['Competition']:.3f}/1.0")
        summary.append(f"  - Accessibility: {top_business['Accessibility']:.3f}/1.0")
        summary.append(f"  - Infrastructure: {top_business['Infrastructure']:.3f}/1.0")
        summary.append(f"  - Local Competitors: {int(top_business['Local Competitors'])}")
        
        summary.append("")
        summary.append("=" * 60)
        
        return "\n".join(summary)


def optimize_location(initial_point: tuple, scoring_fn, step: float = 0.0005, 
                     iterations: int = 300) -> tuple:
    """
    Find optimal business location using gradient-free optimization.
    Uses local search/hill climbing with random perturbations.
    
    Args:
        initial_point: Starting (lat, lon) tuple
        scoring_fn: Function that takes (lat, lon) and returns score
        step: Maximum perturbation distance per iteration
        iterations: Number of optimization iterations
        
    Returns:
        Tuple of (optimal_point, best_score)
    """
    best = np.array(initial_point)
    best_score = scoring_fn(best)
    
    for i in range(iterations):
        # Random perturbation
        perturbation = np.random.uniform(-step, step, size=2)
        candidate = best + perturbation
        
        score = scoring_fn(candidate)
        
        # Accept if better
        if score > best_score:
            best = candidate
            best_score = score
    
    return tuple(best), best_score


def predict_counterfactual_effect(point: tuple, business_type: str, 
                                 features: dict) -> float:
    """
    Predict the impact of opening a business at a location.
    Counterfactual: "What if we open here?"
    
    Args:
        point: (lat, lon) coordinate
        business_type: Type of business
        features: Dictionary with feature values (footfall, competition, transit, etc.)
        
    Returns:
        Predicted change in local market score
    """
    predicted_change = (
        features.get("footfall", 0) * 0.1 -
        features.get("competition", 0) * 0.05 +
        features.get("transit", 0) * 0.02
    )
    
    return predicted_change
