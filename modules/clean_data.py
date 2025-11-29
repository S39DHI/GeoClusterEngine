"""
Data Cleaning Module
Handles data cleaning, preparation, and validation for POI data
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


class DataCleaner:
    """
    Handles cleaning and preparation of POI data including:
    - Removing duplicates
    - Fixing invalid coordinates
    - Normalizing category labels
    - Converting data formats
    """
    
    CATEGORY_MAPPING = {
        'cafe': 'cafe',
        'coffee': 'cafe',
        'coffee_shop': 'cafe',
        'restaurant': 'restaurant',
        'fast_food': 'restaurant',
        'food_court': 'restaurant',
        'shop': 'retail',
        'supermarket': 'retail',
        'convenience': 'retail',
        'mall': 'retail',
        'bank': 'finance',
        'atm': 'finance',
        'hospital': 'healthcare',
        'pharmacy': 'healthcare',
        'clinic': 'healthcare',
        'school': 'education',
        'university': 'education',
        'college': 'education',
        'hotel': 'hospitality',
        'hostel': 'hospitality',
        'motel': 'hospitality'
    }
    
    def __init__(self):
        """Initialize the DataCleaner."""
        self.cleaned_data = None
        self.cleaning_stats = {}
    
    def clean_geodataframe(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Main cleaning function for GeoDataFrame.
        
        Args:
            gdf: Input GeoDataFrame with POI data
            
        Returns:
            Cleaned GeoDataFrame
        """
        if len(gdf) == 0:
            return gdf
            
        initial_count = len(gdf)
        self.cleaning_stats['initial_count'] = initial_count
        
        gdf = self._extract_coordinates(gdf)
        
        gdf = self._remove_invalid_coordinates(gdf)
        self.cleaning_stats['after_coord_validation'] = len(gdf)
        
        gdf = self._remove_duplicates(gdf)
        self.cleaning_stats['after_dedup'] = len(gdf)
        
        gdf = self._normalize_categories(gdf)
        
        gdf = self._add_derived_fields(gdf)
        
        gdf = self._fill_missing_values(gdf)
        
        self.cleaned_data = gdf
        self.cleaning_stats['final_count'] = len(gdf)
        
        return gdf
    
    def _extract_coordinates(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extract latitude and longitude from geometry.
        
        Args:
            gdf: GeoDataFrame with geometry column
            
        Returns:
            GeoDataFrame with lat/lon columns
        """
        if 'geometry' in gdf.columns:
            gdf['latitude'] = gdf.geometry.apply(
                lambda g: g.centroid.y if g and not g.is_empty else None
            )
            gdf['longitude'] = gdf.geometry.apply(
                lambda g: g.centroid.x if g and not g.is_empty else None
            )
        return gdf
    
    def _remove_invalid_coordinates(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove rows with invalid or missing coordinates.
        
        Args:
            gdf: GeoDataFrame with lat/lon columns
            
        Returns:
            Cleaned GeoDataFrame
        """
        if 'latitude' not in gdf.columns or 'longitude' not in gdf.columns:
            return gdf
            
        valid_mask = (
            gdf['latitude'].notna() &
            gdf['longitude'].notna() &
            (gdf['latitude'] >= -90) &
            (gdf['latitude'] <= 90) &
            (gdf['longitude'] >= -180) &
            (gdf['longitude'] <= 180)
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"Removed {invalid_count} rows with invalid coordinates")
            
        return gdf[valid_mask].copy()
    
    def _remove_duplicates(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove duplicate POIs based on location and name.
        
        Args:
            gdf: GeoDataFrame
            
        Returns:
            Deduplicated GeoDataFrame
        """
        if 'latitude' in gdf.columns and 'longitude' in gdf.columns:
            gdf['lat_round'] = gdf['latitude'].round(5)
            gdf['lon_round'] = gdf['longitude'].round(5)
            
            if 'name' in gdf.columns:
                gdf_dedup = gdf.drop_duplicates(
                    subset=['lat_round', 'lon_round', 'name'],
                    keep='first'
                )
            else:
                gdf_dedup = gdf.drop_duplicates(
                    subset=['lat_round', 'lon_round'],
                    keep='first'
                )
            
            gdf_dedup = gdf_dedup.drop(columns=['lat_round', 'lon_round'])
            
            removed = len(gdf) - len(gdf_dedup)
            if removed > 0:
                print(f"Removed {removed} duplicate POIs")
                
            return gdf_dedup
            
        return gdf
    
    def _normalize_categories(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Normalize category labels to standard format.
        
        Args:
            gdf: GeoDataFrame with category column
            
        Returns:
            GeoDataFrame with normalized categories
        """
        if 'category' in gdf.columns:
            gdf['category'] = gdf['category'].str.lower().str.strip()
            
            gdf['category_group'] = gdf['category'].map(
                lambda x: self.CATEGORY_MAPPING.get(x, 'other')
            )
        
        return gdf
    
    def _add_derived_fields(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add derived fields for analysis.
        
        Args:
            gdf: GeoDataFrame
            
        Returns:
            GeoDataFrame with additional fields
        """
        if 'rating' not in gdf.columns:
            gdf['rating'] = np.nan
            
        if 'footfall_estimate' not in gdf.columns:
            gdf['footfall_estimate'] = np.nan
            
        if 'population_nearby' not in gdf.columns:
            gdf['population_nearby'] = np.nan
        
        if 'geometry' in gdf.columns:
            gdf['point_geometry'] = gdf.geometry.apply(
                lambda g: g.centroid if g else None
            )
        
        return gdf
    
    def _fill_missing_values(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Fill missing values with appropriate defaults.
        
        Args:
            gdf: GeoDataFrame
            
        Returns:
            GeoDataFrame with filled values
        """
        if 'name' in gdf.columns:
            gdf['name'] = gdf['name'].fillna('Unknown')
            gdf['name'] = gdf['name'].replace('', 'Unknown')
        
        if 'source' not in gdf.columns:
            gdf['source'] = 'osm'
            
        return gdf
    
    def csv_to_geodataframe(self, csv_path: str) -> gpd.GeoDataFrame:
        """
        Convert CSV file to GeoDataFrame.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            GeoDataFrame
        """
        df = pd.read_csv(csv_path)
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
        
        geometry = [
            Point(lon, lat) 
            for lon, lat in zip(df['longitude'], df['latitude'])
        ]
        
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        return gdf
    
    def get_cleaning_report(self) -> dict:
        """
        Get a report of the cleaning operations performed.
        
        Returns:
            Dictionary with cleaning statistics
        """
        if not self.cleaning_stats:
            return {"message": "No cleaning performed yet"}
            
        report = {
            "initial_count": self.cleaning_stats.get('initial_count', 0),
            "removed_invalid_coords": (
                self.cleaning_stats.get('initial_count', 0) - 
                self.cleaning_stats.get('after_coord_validation', 0)
            ),
            "removed_duplicates": (
                self.cleaning_stats.get('after_coord_validation', 0) - 
                self.cleaning_stats.get('after_dedup', 0)
            ),
            "final_count": self.cleaning_stats.get('final_count', 0),
            "retention_rate": (
                self.cleaning_stats.get('final_count', 0) / 
                max(self.cleaning_stats.get('initial_count', 1), 1) * 100
            )
        }
        
        return report
    
    def prepare_for_clustering(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Prepare coordinate data for clustering algorithms.
        
        Args:
            gdf: Cleaned GeoDataFrame
            
        Returns:
            Numpy array of coordinates (lat, lon)
        """
        if 'latitude' not in gdf.columns or 'longitude' not in gdf.columns:
            raise ValueError("GeoDataFrame must have latitude and longitude columns")
            
        coords = gdf[['latitude', 'longitude']].values
        
        valid_mask = ~np.isnan(coords).any(axis=1)
        coords = coords[valid_mask]
        
        return coords
    
    def filter_by_category(self, gdf: gpd.GeoDataFrame, 
                           categories: list) -> gpd.GeoDataFrame:
        """
        Filter GeoDataFrame by categories.
        
        Args:
            gdf: Input GeoDataFrame
            categories: List of categories to keep
            
        Returns:
            Filtered GeoDataFrame
        """
        if 'category' not in gdf.columns:
            return gdf
            
        categories_lower = [c.lower() for c in categories]
        mask = gdf['category'].str.lower().isin(categories_lower)
        
        return gdf[mask].copy()
    
    def filter_by_bounds(self, gdf: gpd.GeoDataFrame,
                         min_lat: float, max_lat: float,
                         min_lon: float, max_lon: float) -> gpd.GeoDataFrame:
        """
        Filter GeoDataFrame by geographic bounds.
        
        Args:
            gdf: Input GeoDataFrame
            min_lat, max_lat: Latitude bounds
            min_lon, max_lon: Longitude bounds
            
        Returns:
            Filtered GeoDataFrame
        """
        mask = (
            (gdf['latitude'] >= min_lat) &
            (gdf['latitude'] <= max_lat) &
            (gdf['longitude'] >= min_lon) &
            (gdf['longitude'] <= max_lon)
        )
        
        return gdf[mask].copy()
