# BizWiz

A machine learning-powered business location optimization system that leverages external APIs and demographic data to identify optimal locations for new business establishments.

## Purpose and Scope
This document provides a high-level overview of the BizWiz platform, a machine learning-powered business location optimization system that leverages external APIs and demographic data to identify optimal locations for new business establishments. The platform analyzes geographic, demographic, and competitive factors to predict revenue potential and rank locations for business expansion.

## System Overview
BizWiz is a comprehensive platform that combines multiple data sources with machine learning algorithms to provide business location analysis and recommendations. The system uses RandomForestRegressor models to predict revenue potential based on demographic data, competition analysis, foot traffic patterns, and rental market conditions.

### Core System Architecture
<img width="1491" alt="Screenshot 2025-05-28 at 11 16 46 AM" src="https://github.com/user-attachments/assets/5bf66ccd-70f0-4031-8794-c00da7bdbf33" />

## Business-Specific Analysis Systems
The platform includes specialized analysis systems tailored for different business types and geographic scopes:
<img width="734" alt="Screenshot 2025-05-28 at 11 18 34 AM" src="https://github.com/user-attachments/assets/7135e9da-0e1d-4f4e-b263-af209c7f6db2" />
<img width="932" alt="Screenshot 2025-05-28 at 11 19 23 AM" src="https://github.com/user-attachments/assets/d86e43da-ca0e-467c-88e3-ba991570c6d3" />

### Sources: System specialization diagrams, analysis pipeline documentation

## Platform Components
### Core Analysis Engines
The platform consists of five main Jupyter notebook-based analysis engines:
- ShopWizard System: Specialized for barbershop location optimization in the Minneapolis-St. Paul metropolitan area
- Restaurant Location Systems: Multiple implementations for Raising Cane's restaurant chain expansion analysis
- General Business Analysis: Broad-purpose system for various business types with product recommendation capabilities
### Web User Interfaces
Two HTML-based wizard interfaces provide user-friendly access to the analysis systems:
- bizwiz.html: Business wizard interface for general location analysis
- caneswizard.html: Specialized interface for restaurant location analysis

## External Data Integration
The platform integrates with multiple external APIs to gather comprehensive location data:
- Google Maps Places API: Competition and foot traffic data
- US Census Bureau ACS5: demographic and population statistics
- RentCast API: rental market data and property information
- FCC Geo API: geographic coordinate validation and conversion
## Machine Learning Pipeline
All systems utilize a common ML architecture built on RandomForestRegressor models with standardized feature engineering processes for revenue prediction and location ranking.
## Data Flow and Processing
### System Integration and Data Flow
<img width="1366" alt="Screenshot 2025-05-28 at 11 24 13 AM" src="https://github.com/user-attachments/assets/5a6ee160-6422-4195-b027-3b288ca0c67f" />
### Sources: Data flow diagrams, API integration patterns, sequence diagrams

## Technical Architecture Overview
The BizWiz platform follows a modular architecture with shared components across specialized systems:
- Shared ML Core: Common RandomForestRegressor implementation with standardized feature engineering
- API Integration Layer: Unified interface for external data source access with caching mechanisms
- Visualization Framework: Dash and Plotly based interactive mapping and analytics dashboard
- Web Interface Layer: HTML wizard interfaces for user interaction

# Key Location Findings
## 1. 
47.935,-97.08
<img width="1018" alt="1" src="https://github.com/user-attachments/assets/905436c7-181d-4e68-9ab1-fe2fbdcec66d" />
<img width="510" alt="1 2" src="https://github.com/user-attachments/assets/ab15e1c6-960a-43ab-a439-67237645947a" />

## 2. 
47.885,-97.095
<img width="1023" alt="2" src="https://github.com/user-attachments/assets/52a0869f-3314-47f9-b90a-6e1b35d8e3e4" />
<img width="552" alt="2 2" src="https://github.com/user-attachments/assets/ba26ff6f-3b08-47b0-b489-21698d95e59a" />

## 3. 
47.91,-97.085
<img width="999" alt="3 2" src="https://github.com/user-attachments/assets/d3388fea-92dc-4b91-adb6-88dc8ef94d1a" />
<img width="574" alt="3 1" src="https://github.com/user-attachments/assets/1dde8921-0fb9-4f07-9d5a-8edf1dfb45c5" />

## 4. 
47.905,-97.075
<img width="1025" alt="4 2" src="https://github.com/user-attachments/assets/6dc53113-a6f9-4133-b93f-1e141fc0eea4" />
<img width="563" alt="4 1" src="https://github.com/user-attachments/assets/07758847-6221-4032-b1d3-5881b8e515f6" />

## 5. 
47.91,-97.055
<img width="1033" alt="5 2" src="https://github.com/user-attachments/assets/770f87ad-fbb2-4d42-bc89-1540675272ba" />
<img width="677" alt="5 1" src="https://github.com/user-attachments/assets/71373a18-6e77-4e1a-965d-16003d7b7360" />

## City
![newplot](https://github.com/user-attachments/assets/88ada6b2-ed13-483c-bd85-9be7e4eb1e16)


# Improved Code - V2
```
# === IMPORTS ===
import os
import numpy as np
import pandas as pd
import googlemaps
import requests
import time
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv
import pickle
from functools import lru_cache

# === LOAD .env VARIABLES ===
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
RENTCAST_API_KEY = os.getenv('RENTCAST_API_KEY')

# === GOOGLE MAPS CLIENT ===
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# === GRID GENERATION - GRAND FORKS, ND ===
min_lat, max_lat = 47.85, 47.95
min_lon, max_lon = -97.15, -97.0
grid_spacing = 0.005
lats = np.arange(min_lat, max_lat, grid_spacing)
lons = np.arange(min_lon, max_lon, grid_spacing)
grid_points = [(lat, lon) for lat in lats for lon in lons]

# === CACHING SETUP ===
CACHE_FILE = 'location_data_cache.pkl'

def load_cache():
    try:
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

# === DISTANCE FUNCTION IN MILES ===
def calculate_distance_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    c = 2 * asin(sqrt(a))
    return c * 3956

# === BULK DATA FETCHING ===
class LocationDataFetcher:
    def __init__(self):
        self.cache = load_cache()
        self.chickfila_locations = None
        self.competitor_locations = {}
        self.poi_locations = {}
        self.active_listings = []
        
    def fetch_all_chickfila_locations(self):
        """Fetch all Chick-fil-A locations in the broader area once"""
        if self.chickfila_locations is not None:
            return
            
        cache_key = 'chickfila_all'
        if cache_key in self.cache:
            self.chickfila_locations = self.cache[cache_key]
            return
            
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        try:
            # Search in a large radius to get all nearby locations
            result = gmaps.places_nearby(
                location=(center_lat, center_lon), 
                radius=50000,  # 50km radius
                keyword='chick-fil-a'
            )
            locations = result['results']
            
            # Handle pagination if needed
            while 'next_page_token' in result:
                time.sleep(2)
                result = gmaps.places_nearby(
                    location=(center_lat, center_lon),
                    radius=50000,
                    keyword='chick-fil-a',
                    page_token=result['next_page_token']
                )
                locations.extend(result['results'])
                
            self.chickfila_locations = [(
                loc['geometry']['location']['lat'],
                loc['geometry']['location']['lng']
            ) for loc in locations]
            
            self.cache[cache_key] = self.chickfila_locations
            save_cache(self.cache)
            
        except Exception as e:
            print(f"Error fetching Chick-fil-A locations: {e}")
            self.chickfila_locations = []
    
    def fetch_competitor_locations(self):
        """Fetch all competitor locations once"""
        if self.competitor_locations:
            return
            
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        competitors = ['mcdonalds', 'kfc', 'taco bell', 'burger king', 'subway', 'wendys', 'popeyes']
        
        for competitor in competitors:
            cache_key = f'competitor_{competitor}'
            if cache_key in self.cache:
                self.competitor_locations[competitor] = self.cache[cache_key]
                continue
                
            try:
                result = gmaps.places_nearby(
                    location=(center_lat, center_lon),
                    radius=20000,  # 20km radius
                    keyword=competitor
                )
                locations = [(
                    loc['geometry']['location']['lat'],
                    loc['geometry']['location']['lng']
                ) for loc in result['results']]
                
                self.competitor_locations[competitor] = locations
                self.cache[cache_key] = locations
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching {competitor} locations: {e}")
                self.competitor_locations[competitor] = []
        
        save_cache(self.cache)
    
    def fetch_poi_locations(self):
        """Fetch points of interest once"""
        if self.poi_locations:
            return
            
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        poi_types = [
            ('university', 'university', 10),
            ('shopping_mall', 'shopping_mall', 5), 
            ('store', 'store', 2),
            ('restaurant', 'restaurant', 1)
        ]
        
        for poi_name, poi_type, weight in poi_types:
            cache_key = f'poi_{poi_name}'
            if cache_key in self.cache:
                self.poi_locations[poi_name] = self.cache[cache_key]
                continue
                
            try:
                if poi_name == 'university':
                    result = gmaps.places_nearby(
                        location=(center_lat, center_lon),
                        radius=20000,
                        keyword='university'
                    )
                else:
                    result = gmaps.places_nearby(
                        location=(center_lat, center_lon),
                        radius=20000,
                        type=poi_type
                    )
                    
                locations = [(
                    loc['geometry']['location']['lat'],
                    loc['geometry']['location']['lng'],
                    weight
                ) for loc in result['results']]
                
                self.poi_locations[poi_name] = locations
                self.cache[cache_key] = locations
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error fetching {poi_name} locations: {e}")
                self.poi_locations[poi_name] = []
        
        save_cache(self.cache)

    def fetch_rental_listings(self):
        """Fetch rental listings once"""
        if self.active_listings:
            return
            
        cache_key = 'rental_listings'
        if cache_key in self.cache:
            # Check if cache is recent (less than 24 hours old)
            cache_time = self.cache.get(f'{cache_key}_timestamp', 0)
            if time.time() - cache_time < 86400:  # 24 hours
                self.active_listings = self.cache[cache_key]
                return
        
        try:
            url = "https://api.rentcast.io/v1/listings/rental/long-term"
            headers = {"X-Api-Key": RENTCAST_API_KEY}
            params = {"city": "Grand Forks", "state": "ND", "status": "active", "limit": 500}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                self.active_listings = response.json().get('listings', [])
                self.cache[cache_key] = self.active_listings
                self.cache[f'{cache_key}_timestamp'] = time.time()
                save_cache(self.cache)
            else:
                self.active_listings = []
                
        except Exception as e:
            print(f"Error fetching rental listings: {e}")
            self.active_listings = []

    @lru_cache(maxsize=1000)
    def get_demographics_cached(self, lat_rounded, lon_rounded):
        """Cache demographics by rounded coordinates to avoid duplicate census calls"""
        try:
            fcc_url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat_rounded}&longitude={lon_rounded}&format=json"
            fcc_response = requests.get(fcc_url)
            fips = fcc_response.json()['Block']['FIPS'][:11]
            
            url = f"https://api.census.gov/data/2020/acs/acs5?get=B01003_001E,B19013_001E,B01002_001E&for=tract:{fips[5:11]}&in=state:{fips[:2]}+county:{fips[2:5]}&key={CENSUS_API_KEY}"
            data = requests.get(url).json()[1]
            
            return {
                'population': int(data[0]), 
                'median_income': int(data[1]), 
                'median_age': float(data[2])
            }
        except:
            return {'population': 5000, 'median_income': 45000, 'median_age': 28}

    def calculate_features_for_point(self, lat, lon):
        """Calculate all features for a single point using cached data"""
        # Round coordinates for demographic caching (census tracts don't change much)
        lat_rounded = round(lat, 3)
        lon_rounded = round(lon, 3)
        
        # Chick-fil-A proximity
        if self.chickfila_locations:
            distances_to_chickfila = [
                calculate_distance_miles(lat, lon, c_lat, c_lon) 
                for c_lat, c_lon in self.chickfila_locations
            ]
            chick_count = len([d for d in distances_to_chickfila if d <= 5])  # Within 5 miles
            nearest_chickfila = min(distances_to_chickfila) if distances_to_chickfila else 30
        else:
            chick_count, nearest_chickfila = 0, 30
        
        # Fast food competition
        competition_count = 0
        for competitor, locations in self.competitor_locations.items():
            nearby_competitors = [
                1 for c_lat, c_lon in locations 
                if calculate_distance_miles(lat, lon, c_lat, c_lon) <= 2  # Within 2 miles
            ]
            competition_count += len(nearby_competitors)
        
        # Foot traffic score
        foot_traffic = 0
        for poi_type, locations in self.poi_locations.items():
            for p_lat, p_lon, weight in locations:
                if calculate_distance_miles(lat, lon, p_lat, p_lon) <= 1:  # Within 1 mile
                    foot_traffic += weight
        
        # Demographics (cached)
        demographics = self.get_demographics_cached(lat_rounded, lon_rounded)
        
        # Rental data
        nearby_listings = []
        for listing in self.active_listings:
            if listing.get('latitude') and listing.get('longitude'):
                distance = calculate_distance_miles(
                    lat, lon, listing['latitude'], listing['longitude']
                )
                if distance <= 1:  # Within 1 mile
                    nearby_listings.append(listing['price'])
        
        active_listings_count = len(nearby_listings)
        avg_rent = np.mean(nearby_listings) if nearby_listings else 0
        
        # Zoning (randomized for demo)
        import random
        zoning = random.choice([True, False])
        
        return {
            'latitude': lat,
            'longitude': lon,
            'chickfila_count_nearby': chick_count,
            'distance_to_chickfila': nearest_chickfila,
            'fast_food_competition': competition_count,
            'foot_traffic_score': foot_traffic,
            'population': demographics['population'],
            'median_income': demographics['median_income'],
            'median_age': demographics['median_age'],
            'rent_per_sqft': 12.50,  # Fixed for now
            'zoning_compliant': int(zoning),
            'active_listings_within_1_mile': active_listings_count,
            'average_nearby_rent': avg_rent
        }

# === MAIN DATA COLLECTION ===
def collect_location_data():
    fetcher = LocationDataFetcher()
    
    print("Fetching bulk data...")
    fetcher.fetch_all_chickfila_locations()
    print(f"Found {len(fetcher.chickfila_locations)} Chick-fil-A locations")
    
    fetcher.fetch_competitor_locations()
    total_competitors = sum(len(locs) for locs in fetcher.competitor_locations.values())
    print(f"Found {total_competitors} competitor locations")
    
    fetcher.fetch_poi_locations()
    total_pois = sum(len(locs) for locs in fetcher.poi_locations.values())
    print(f"Found {total_pois} points of interest")
    
    fetcher.fetch_rental_listings()
    print(f"Found {len(fetcher.active_listings)} rental listings")
    
    print("Processing grid points...")
    feature_list = []
    
    for idx, (lat, lon) in enumerate(grid_points):
        if idx % 10 == 0:  # Progress update every 10 points
            print(f"Processing {idx+1}/{len(grid_points)}: {lat:.4f}, {lon:.4f}")
        
        features = fetcher.calculate_features_for_point(lat, lon)
        feature_list.append(features)
        
        # Minimal delay to be respectful to APIs
        if idx % 50 == 0:  # Only sleep every 50 points
            time.sleep(0.1)
    
    return pd.DataFrame(feature_list)

# === DATA PROCESSING ===
print("Starting data collection...")
df = collect_location_data()

# Calculate derived features
df['chick_fil_a_advantage'] = np.where(
    (df['distance_to_chickfila'] > 1) & (df['distance_to_chickfila'] < 5), 
    1000 / df['distance_to_chickfila'], 
    0
)
df['youth_factor'] = np.where(df['median_age'] < 30, 1.5, 1.0)

# Revenue calculation
df['estimated_revenue'] = (
    df['population'] * 0.4 +
    df['median_income'] * 0.0003 +
    df['foot_traffic_score'] * 200 +
    df['chick_fil_a_advantage'] * 500 +
    df['youth_factor'] * 1000 +
    df['active_listings_within_1_mile'] * 100 -
    df['fast_food_competition'] * 300 -
    df['average_nearby_rent'] * 0.1 -
    df['rent_per_sqft'] * 100
)
df['estimated_revenue'] = np.maximum(df['estimated_revenue'], 0)

# Model training
X = df.drop(columns=['latitude', 'longitude', 'estimated_revenue'])
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
y = df['estimated_revenue']

model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
model.fit(X, y)
df['predicted_revenue'] = model.predict(X)

print(f"Data collection complete. Processed {len(df)} locations.")

# === DASH APP ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Optimal Raising Cane's Locations in Grand Forks, ND"), className="text-center my-4")),
    dbc.Row([
        dbc.Col([
            html.Label("Minimum Predicted Revenue:"),
            dcc.Slider(id='revenue-slider', min=0, max=df['predicted_revenue'].max(), step=500, value=df['predicted_revenue'].quantile(0.7)),
            html.Label("Maximum Distance to Chick-fil-A (miles):"),
            dcc.Slider(id='chickfila-distance-slider', min=0, max=30, step=1, value=5),
            html.Label("Minimum Foot Traffic Score:"),
            dcc.Slider(id='traffic-slider', min=0, max=df['foot_traffic_score'].max(), step=5, value=0),
            html.Label("Zoning Compliance:"),
            dcc.RadioItems(id='zoning-radio', options=[{'label': 'All Locations', 'value': 'all'}, {'label': 'Only Compliant', 'value': 'compliant'}], value='compliant'),
            html.Div(id='location-stats', className="mt-4 p-3 bg-light rounded")
        ], width=3),
        dbc.Col(dcc.Graph(id='revenue-map', style={'height': '80vh'}), width=9)
    ])
], fluid=True)

@app.callback(
    [Output('revenue-map', 'figure'), Output('location-stats', 'children')],
    [Input('revenue-slider', 'value'), Input('chickfila-distance-slider', 'value'), Input('traffic-slider', 'value'), Input('zoning-radio', 'value')]
)
def update_map(min_revenue, max_chickfila_distance, min_traffic, zoning_filter):
    filtered = df[
        (df['predicted_revenue'] >= min_revenue) &
        (df['distance_to_chickfila'] <= max_chickfila_distance) &
        (df['foot_traffic_score'] >= min_traffic)
    ]
    if zoning_filter == 'compliant':
        filtered = filtered[filtered['zoning_compliant'] == 1]
    
    fig = px.scatter_mapbox(
        filtered, lat='latitude', lon='longitude', size='predicted_revenue', color='predicted_revenue',
        color_continuous_scale='RdYlGn', size_max=20, zoom=12, mapbox_style='carto-positron'
    )
    
    if len(filtered) > 0:
        best = filtered.loc[filtered['predicted_revenue'].idxmax()]
        stats = html.Div([
            html.H5("Top Location"),
            html.P(f"Latitude: {best['latitude']:.4f}"),
            html.P(f"Longitude: {best['longitude']:.4f}"),
            html.P(f"Predicted Revenue: ${best['predicted_revenue']:,.0f}"),
            html.P(f"Distance to Chick-fil-A: {best['distance_to_chickfila']:.1f} miles"),
            html.P(f"Competition: {best['fast_food_competition']}"),
            html.P(f"Foot Traffic: {best['foot_traffic_score']}"),
            html.P(f"Active Listings (1 mile): {best['active_listings_within_1_mile']}"),
            html.P(f"Average Rent Nearby: ${best['average_nearby_rent']:,.0f}")
        ])
    else:
        stats = html.P("No locations match your filters.")
    
    return fig, stats

if __name__ == '__main__':
    app.run(debug=True)
```


### Code V1 - WARNING HAS EXCESSIVE API CALLS BUILT IN
```
# === IMPORTS ===
import os
import numpy as np
import pandas as pd
import googlemaps
import requests
import time
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

# === LOAD .env VARIABLES ===
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
RENTCAST_API_KEY = os.getenv('RENTCAST_API_KEY')

# === GOOGLE MAPS CLIENT ===
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# === GRID GENERATION - GRAND FORKS, ND ===
min_lat, max_lat = 47.85, 47.95
min_lon, max_lon = -97.15, -97.0
grid_spacing = 0.005
lats = np.arange(min_lat, max_lat, grid_spacing)
lons = np.arange(min_lon, max_lon, grid_spacing)
grid_points = [(lat, lon) for lat in lats for lon in lons]

# === DISTANCE FUNCTION IN MILES ===
def calculate_distance_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    c = 2 * asin(sqrt(a))
    return c * 3956

# === RENTCAST LISTINGS ===
def fetch_active_rental_listings(city, state, api_key):
    url = "https://api.rentcast.io/v1/listings/rental/long-term"
    headers = {"X-Api-Key": api_key}
    params = {"city": city, "state": state, "status": "active", "limit": 500}
    response = requests.get(url, headers=headers, params=params)
    return response.json().get('listings', []) if response.status_code == 200 else []

active_listings = fetch_active_rental_listings("Grand Forks", "ND", RENTCAST_API_KEY)

# === GOOGLE MAPS FEATURES ===
def get_chickfila_proximity(lat, lon, radius=8000):
    try:
        result = gmaps.places_nearby(location=(lat, lon), radius=radius, keyword='chick-fil-a')
        chickfila_locations = result['results']
        while 'next_page_token' in result:
            time.sleep(2)
            result = gmaps.places_nearby(location=(lat, lon), radius=radius, keyword='chick-fil-a', page_token=result['next_page_token'])
            chickfila_locations.extend(result['results'])
        count = len(chickfila_locations)
        distances = [calculate_distance_miles(lat, lon, p['geometry']['location']['lat'], p['geometry']['location']['lng']) for p in chickfila_locations]
        nearest_distance = min(distances) if distances else 30
        return count, nearest_distance
    except:
        return 0, 30

def get_fast_food_competition(lat, lon, radius=3200):
    try:
        competitors = ['mcdonalds', 'kfc', 'taco bell', 'burger king', 'subway', 'wendys', 'popeyes']
        total = 0
        for c in competitors:
            total += len(gmaps.places_nearby(location=(lat, lon), radius=radius, keyword=c)['results'])
            time.sleep(0.2)
        return total
    except:
        return 0

def get_foot_traffic(lat, lon, radius=1600):
    try:
        u = gmaps.places_nearby(location=(lat, lon), radius=radius, keyword='university')
        s = gmaps.places_nearby(location=(lat, lon), radius=radius, type='shopping_mall')
        st = gmaps.places_nearby(location=(lat, lon), radius=radius, type='store')
        r = gmaps.places_nearby(location=(lat, lon), radius=radius, type='restaurant')
        return len(u['results'])*10 + len(s['results'])*5 + len(st['results'])*2 + len(r['results'])
    except:
        return 0

def get_demographics(lat, lon):
    try:
        fcc_url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"
        fips = requests.get(fcc_url).json()['Block']['FIPS'][:11]
        url = f"https://api.census.gov/data/2020/acs/acs5?get=B01003_001E,B19013_001E,B01002_001E&for=tract:{fips[5:11]}&in=state:{fips[:2]}+county:{fips[2:5]}&key={CENSUS_API_KEY}"
        data = requests.get(url).json()[1]
        return {'population': int(data[0]), 'median_income': int(data[1]), 'median_age': float(data[2])}
    except:
        return {'population': 5000, 'median_income': 45000, 'median_age': 28}

def check_zoning(lat, lon):
    import random
    return random.choice([True, False])

# === DATA COLLECTION ===
feature_list = []
for idx, (lat, lon) in enumerate(grid_points):
    print(f"Processing {idx+1}/{len(grid_points)}: {lat:.4f}, {lon:.4f}")
    chick_count, chick_dist = get_chickfila_proximity(lat, lon)
    comp = get_fast_food_competition(lat, lon)
    traffic = get_foot_traffic(lat, lon)
    demo = get_demographics(lat, lon)
    rent = 12.50
    zoning = check_zoning(lat, lon)

    distances = [calculate_distance_miles(lat, lon, listing['latitude'], listing['longitude']) for listing in active_listings if listing.get('latitude') and listing.get('longitude')]
    rents = [listing['price'] for i, listing in enumerate(active_listings) if listing.get('latitude') and listing.get('longitude') and distances[i] <= 1]
    active_listings_count = len([d for d in distances if d <= 1])
    avg_rent = np.mean(rents) if rents else 0

    feature_list.append({
        'latitude': lat, 'longitude': lon,
        'chickfila_count_nearby': chick_count,
        'distance_to_chickfila': chick_dist,
        'fast_food_competition': comp,
        'foot_traffic_score': traffic,
        'population': demo['population'],
        'median_income': demo['median_income'],
        'median_age': demo['median_age'],
        'rent_per_sqft': rent,
        'zoning_compliant': int(zoning),
        'active_listings_within_1_mile': active_listings_count,
        'average_nearby_rent': avg_rent
    })
    time.sleep(0.3)

df = pd.DataFrame(feature_list)
df['chick_fil_a_advantage'] = np.where((df['distance_to_chickfila'] > 1) & (df['distance_to_chickfila'] < 5), 1000 / df['distance_to_chickfila'], 0)
df['youth_factor'] = np.where(df['median_age'] < 30, 1.5, 1.0)
df['estimated_revenue'] = (
    df['population'] * 0.4 +
    df['median_income'] * 0.0003 +
    df['foot_traffic_score'] * 200 +
    df['chick_fil_a_advantage'] * 500 +
    df['youth_factor'] * 1000 +
    df['active_listings_within_1_mile'] * 100 -
    df['fast_food_competition'] * 300 -
    df['average_nearby_rent'] * 0.1 -
    df['rent_per_sqft'] * 100
)
df['estimated_revenue'] = np.maximum(df['estimated_revenue'], 0)

X = df.drop(columns=['latitude', 'longitude', 'estimated_revenue'])
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
y = df['estimated_revenue']

model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
model.fit(X, y)
df['predicted_revenue'] = model.predict(X)

# === DASH APP ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Optimal Raising Cane's Locations in Grand Forks, ND"), className="text-center my-4")),
    dbc.Row([
        dbc.Col([
            html.Label("Minimum Predicted Revenue:"),
            dcc.Slider(id='revenue-slider', min=0, max=df['predicted_revenue'].max(), step=500, value=df['predicted_revenue'].quantile(0.7)),
            html.Label("Maximum Distance to Chick-fil-A (miles):"),
            dcc.Slider(id='chickfila-distance-slider', min=0, max=30, step=1, value=5),
            html.Label("Minimum Foot Traffic Score:"),
            dcc.Slider(id='traffic-slider', min=0, max=df['foot_traffic_score'].max(), step=5, value=0),
            html.Label("Zoning Compliance:"),
            dcc.RadioItems(id='zoning-radio', options=[{'label': 'All Locations', 'value': 'all'}, {'label': 'Only Compliant', 'value': 'compliant'}], value='compliant'),
            html.Div(id='location-stats', className="mt-4 p-3 bg-light rounded")
        ], width=3),
        dbc.Col(dcc.Graph(id='revenue-map', style={'height': '80vh'}), width=9)
    ])
], fluid=True)

@app.callback(
    [Output('revenue-map', 'figure'), Output('location-stats', 'children')],
    [Input('revenue-slider', 'value'), Input('chickfila-distance-slider', 'value'), Input('traffic-slider', 'value'), Input('zoning-radio', 'value')]
)
def update_map(min_revenue, max_chickfila_distance, min_traffic, zoning_filter):
    filtered = df[
        (df['predicted_revenue'] >= min_revenue) &
        (df['distance_to_chickfila'] <= max_chickfila_distance) &
        (df['foot_traffic_score'] >= min_traffic)
    ]
    if zoning_filter == 'compliant':
        filtered = filtered[filtered['zoning_compliant'] == 1]
    fig = px.scatter_mapbox(
        filtered, lat='latitude', lon='longitude', size='predicted_revenue', color='predicted_revenue',
        color_continuous_scale='RdYlGn', size_max=20, zoom=12, mapbox_style='carto-positron'
    )
    if len(filtered) > 0:
        best = filtered.loc[filtered['predicted_revenue'].idxmax()]
        stats = html.Div([
            html.H5("Top Location"),
            html.P(f"Latitude: {best['latitude']:.4f}"),
            html.P(f"Longitude: {best['longitude']:.4f}"),
            html.P(f"Predicted Revenue: ${best['predicted_revenue']:,.0f}"),
            html.P(f"Distance to Chick-fil-A: {best['distance_to_chickfila']:.1f} miles"),
            html.P(f"Competition: {best['fast_food_competition']}"),
            html.P(f"Foot Traffic: {best['foot_traffic_score']}"),
            html.P(f"Active Listings (1 mile): {best['active_listings_within_1_mile']}"),
            html.P(f"Average Rent Nearby: ${best['average_nearby_rent']:,.0f}")
        ])
    else:
        stats = html.P("No locations match your filters.")
    return fig, stats

if __name__ == '__main__':
    app.run(debug=True)
```



View as webpage @ https://nbergeland.github.io/BizWiz/ & https://deepwiki.com/nbergeland/BizWiz/1-bizwiz-platform-overview
