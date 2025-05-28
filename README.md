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
Sources: System specialization diagrams, analysis pipeline documentation

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

```
import numpy as np
import pandas as pd
import googlemaps
import requests
import time
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from math import radians, cos, sin, asin, sqrt

# === API KEYS ===

GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY_HERE'
# Replace with your actual Google API key
CENSUS_API_KEY = 'YOUR_CENSUS_API_KEY_HERE'
# Replace with your actual Census API key
RENTCAST_API_KEY = 'YOUR_RENTCAST_API_KEY_HERE'
# Replace with your actual Rentcast API key

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
