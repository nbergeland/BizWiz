# === DASH APP ===
import os
import numpy as np
import pandas as pd
import googlemaps
import requests
import time
import json
import datetime
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv
import pickle
from functools import lru_cache
import pgeocode

# === LOAD .env VARIABLES ===
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
RENTCAST_API_KEY = os.getenv('RENTCAST_API_KEY')

# === GOOGLE MAPS CLIENT ===
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# === ZIP CODE UTILITIES ===
def get_zip_coordinates(zip_code):
    """Get coordinates for a zip code using pgeocode with Google fallback"""
    try:
        nomi = pgeocode.Nominatim('us')
        location = nomi.query_postal_code(zip_code)
        if pd.notna(location.latitude) and pd.notna(location.longitude):
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Error getting coordinates for zip {zip_code} with pgeocode: {e}")

    try:
        track_api_call()
        geocode_result = gmaps.geocode(zip_code)
        if geocode_result:
            loc = geocode_result[0]['geometry']['location']
            return loc['lat'], loc['lng']
    except Exception as e:
        print(f"Error with fallback geocoding for zip {zip_code}: {e}")

    return None, None

# Global variable to store analysis data
analysis_data = {'df': None, 'bounds': None}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Optimal Raising Cane's Location Analyzer"), className="text-center my-4")),

    # Zip Code Input Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Enter Analysis Area", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("First Zip Code:"),
                            dbc.Input(id='zip1-input', type='text', placeholder='12345', maxlength=5)
                        ], width=6),
                        dbc.Col([
                            html.Label("Second Zip Code:"),
                            dbc.Input(id='zip2-input', type='text', placeholder='67890', maxlength=5)
                        ], width=6),
                    ]),
                    html.Br(),
                    dbc.Button("Analyze Area", id='analyze-button', color='primary', className="w-100"),
                    html.Div(id='analysis-status', className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    # Analysis Controls and Map
    html.Div(id='analysis-content', children=[
        dbc.Row([
            dbc.Col([
                html.Label("Minimum Predicted Revenue:"),
                dcc.Slider(id='revenue-slider', min=0, max=100000, step=1000, value=30000),

                html.Label("Maximum Distance to Chick-fil-A (miles):"),
                dcc.Slider(id='chickfila-distance-slider', min=0, max=15, step=1, value=8),

                html.Label("Minimum Commercial Traffic Score:"),
                dcc.Slider(id='commercial-traffic-slider', min=0, max=200, step=10, value=20),

                html.Label("Maximum Fast Food Competition:"),
                dcc.Slider(id='competition-slider', min=0, max=15, step=1, value=8),

                html.Label("Zoning Compliance:"),
                dcc.RadioItems(
                    id='zoning-radio', 
                    options=[
                        {'label': 'All Locations', 'value': 'all'}, 
                        {'label': 'Only Compliant', 'value': 'compliant'}
                    ], 
                    value='compliant'
                ),

                html.Div(id='location-stats', className="mt-4 p-3 bg-light rounded")
            ], width=3),

            dbc.Col([
                dcc.Graph(id='revenue-map', style={'height': '80vh'})
            ], width=9)
        ])
    ], style={'display': 'none'})  # Hidden initially
], fluid=True)

@app.callback(
    [Output('analysis-status', 'children'),
     Output('analysis-content', 'style'),
     Output('revenue-slider', 'max'),
     Output('revenue-slider', 'value'),
     Output('commercial-traffic-slider', 'max')],
    [Input('analyze-button', 'n_clicks')],
    [State('zip1-input', 'value'),
     State('zip2-input', 'value')]
)
def analyze_zip_codes(n_clicks, zip1, zip2):
    if n_clicks is None:
        return "", {'display': 'none'}, 100000, 30000, 200

    # Validate zip codes
    if not zip1 or not zip2 or len(zip1) != 5 or len(zip2) != 5:
        return dbc.Alert("Please enter two valid 5-digit zip codes.", color="danger"), {'display': 'none'}, 100000, 30000, 200

    try:
        # Show loading message
        loading_msg = dbc.Alert("Analyzing area... This may take a few minutes.", color="info")

        # Perform analysis
        df_result, bounds = collect_commercial_location_data(zip1, zip2)

        if df_result is None:
            return dbc.Alert("Error analyzing the specified area. Please check.", color="danger"), {'display': 'none'}, 100000, 30000, 200

        analysis_data['df'] = df_result
        analysis_data['bounds'] = bounds

        max_revenue = int(df_result['predicted_revenue'].max())
        max_traffic = int(df_result['commercial_traffic_score'].max())

        return dbc.Alert("Analysis complete. Use filters below to refine results.", color="success"), {'display': 'block'}, max_revenue, int(max_revenue * 0.3), max_traffic

    except Exception as e:
        print(f"Error during analysis: {e}")
        return dbc.Alert("Unexpected error occurred. Please try again later.", color="danger"), {'display': 'none'}, 100000, 30000, 200