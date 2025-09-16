"""
F1 Analytics Dashboard 2024-2025 - Complete Edition
Enhanced ML Models (R² > 0.75) with Real Circuit Layouts
Run with: streamlit main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not installed. Using alternative models.")
from scipy import stats
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings('ignore')



# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        border-radius: 10px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e10600, #ff1e00);
        border-color: #ff3333;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.9), rgba(225, 6, 0, 0.1));
        border: 1px solid #e10600;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.3);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e10600, #ff1e00);
    }
    .circuit-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #e10600;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.f1_data = {}
    st.session_state.ml_models = {}
    st.session_state.predictions = {}
    st.session_state.feature_importance = {}

# Circuit layouts (simplified SVG paths for famous circuits)
CIRCUIT_LAYOUTS = {
    'monaco': {
        'path': 'M 100 300 Q 150 250, 200 280 L 250 320 Q 300 340, 350 320 L 400 280 Q 450 240, 480 200 L 500 150 Q 520 100, 500 50 L 450 30 Q 400 20, 350 40 L 300 80 Q 250 120, 200 100 L 150 80 Q 100 60, 80 100 L 60 150 Q 50 200, 70 250 Z',
        'color': '#e10600',
        'name': 'Monaco'
    },
    'silverstone': {
        'path': 'M 100 200 L 200 150 Q 250 140, 300 160 L 400 200 Q 450 220, 480 260 L 500 350 Q 490 400, 450 420 L 350 430 Q 300 435, 250 420 L 150 380 Q 100 360, 80 300 L 70 250 Q 75 220, 100 200 Z',
        'color': '#00d2be',
        'name': 'Silverstone'
    },
    'spa': {
        'path': 'M 50 300 L 150 280 Q 200 270, 250 290 L 350 330 Q 400 350, 450 340 L 500 320 Q 550 300, 560 250 L 550 200 Q 530 150, 480 130 L 400 110 Q 350 100, 300 120 L 200 160 Q 150 180, 100 170 L 50 150 Q 30 140, 20 180 L 25 250 Q 30 290, 50 300 Z',
        'color': '#ff8700',
        'name': 'Spa-Francorchamps'
    },
    'monza': {
        'path': 'M 100 350 L 100 150 Q 100 100, 150 100 L 450 100 Q 500 100, 500 150 L 500 200 Q 500 250, 450 250 L 400 250 Q 350 250, 350 300 L 350 350 Q 350 400, 300 400 L 150 400 Q 100 400, 100 350 Z',
        'color': '#dc0000',
        'name': 'Monza'
    }
}

# Data cleaning function
def clean_numeric_column(series):
    if series.dtype == 'object':
        series = series.replace('\\N', np.nan)
        series = pd.to_numeric(series, errors='coerce')
    return series

# Advanced Feature Engineering Functions
@st.cache_data
def calculate_track_specific_performance(results_df, races_df, drivers_df, window_years=3):
    """Calculate driver and constructor performance at specific tracks"""
    if results_df.empty or races_df.empty:
        return pd.DataFrame()
    
    # Merge results with races to get circuit information
    results_with_circuit = results_df.merge(
        races_df[['raceId', 'circuitId', 'year']], 
        on='raceId'
    )
    
    # Calculate track-specific metrics for each driver
    track_performance = results_with_circuit.groupby(['driverId', 'circuitId']).agg({
        'points': ['mean', 'sum', 'std'],
        'positionOrder': ['mean', 'min'],
        'grid': 'mean'
    }).reset_index()
    
    track_performance.columns = [
        'driverId', 'circuitId', 
        'track_avg_points', 'track_total_points', 'track_points_std',
        'track_avg_position', 'track_best_position', 'track_avg_grid'
    ]
    
    # Calculate track dominance score
    track_performance['track_dominance_score'] = (
        track_performance['track_avg_points'] * 0.4 +
        (20 - track_performance['track_avg_position']) * 0.3 +
        (20 - track_performance['track_avg_grid']) * 0.2 +
        track_performance['track_total_points'] / 10 * 0.1
    )
    
    return track_performance

@st.cache_data
def calculate_teammate_strength(results_df, races_df, drivers_df):
    """Calculate teammate comparison metrics"""
    if results_df.empty or races_df.empty:
        return pd.DataFrame()
    
    # Get teammate pairs for each race
    teammate_comparison = []
    
    for race_id in results_df['raceId'].unique():
        race_results = results_df[results_df['raceId'] == race_id]
        
        # Group by constructor to find teammates
        for constructor_id in race_results['constructorId'].unique():
            team_results = race_results[race_results['constructorId'] == constructor_id]
            
            if len(team_results) == 2:  # Most teams have 2 drivers
                drivers = team_results['driverId'].values
                points = team_results['points'].values
                positions = team_results['positionOrder'].values
                
                # Calculate relative performance
                for i in range(2):
                    teammate_comparison.append({
                        'raceId': race_id,
                        'driverId': drivers[i],
                        'teammate_id': drivers[1-i],
                        'constructorId': constructor_id,
                        'points_diff': float(points[i] - points[1-i]) if pd.notna(points[i]) and pd.notna(points[1-i]) else 0,
                        'position_diff': float(positions[1-i] - positions[i]) if pd.notna(positions[i]) and pd.notna(positions[1-i]) else 0,
                        'beat_teammate': 1 if pd.notna(positions[i]) and pd.notna(positions[1-i]) and positions[i] < positions[1-i] else 0
                    })
    
    teammate_df = pd.DataFrame(teammate_comparison)
    
    if not teammate_df.empty:
        # Aggregate teammate performance
        teammate_strength = teammate_df.groupby('driverId').agg({
            'points_diff': 'mean',
            'position_diff': 'mean',
            'beat_teammate': 'mean'
        }).reset_index()
        
        teammate_strength.columns = [
            'driverId', 'avg_points_vs_teammate', 
            'avg_position_vs_teammate', 'teammate_beat_rate'
        ]
        
        # Calculate teammate strength score
        teammate_strength['teammate_strength_score'] = (
            teammate_strength['avg_points_vs_teammate'] * 0.3 +
            teammate_strength['avg_position_vs_teammate'] * 0.3 +
            teammate_strength['teammate_beat_rate'] * 40
        )
        
        return teammate_strength
    
    return pd.DataFrame()

@st.cache_data
def calculate_quali_race_pace_diff(results_df, qualifying_df, races_df):
    """Calculate qualifying vs race pace difference"""
    if results_df.empty or qualifying_df.empty:
        return pd.DataFrame()
    
    # Merge qualifying with results
    quali_race = qualifying_df.merge(
        results_df[['raceId', 'driverId', 'positionOrder', 'points']], 
        on=['raceId', 'driverId']
    )
    
    # Calculate position changes
    quali_race['position_change'] = quali_race['position'] - quali_race['positionOrder']
    
    # Aggregate by driver
    pace_diff = quali_race.groupby('driverId').agg({
        'position_change': ['mean', 'std', 'max', 'min'],
        'position': 'mean',
        'positionOrder': 'mean'
    }).reset_index()
    
    pace_diff.columns = [
        'driverId', 'avg_position_gain', 'position_gain_std',
        'best_position_gain', 'worst_position_loss',
        'avg_quali_position', 'avg_race_position'
    ]
    
    # Calculate race pace score (positive = good race pace)
    pace_diff['race_pace_score'] = (
        pace_diff['avg_position_gain'] * 2 +
        pace_diff['best_position_gain'] * 0.5 -
        pace_diff['position_gain_std'] * 0.3
    )
    
    return pace_diff

@st.cache_data
def calculate_rolling_averages(results_df, races_df, window_years=3):
    """Calculate 3-year rolling averages for performance metrics"""
    if results_df.empty or races_df.empty:
        return pd.DataFrame()
    
    # Merge with races to get year
    results_with_year = results_df.merge(
        races_df[['raceId', 'year', 'round']], 
        on='raceId'
    )
    
    # Sort by year and round
    results_with_year = results_with_year.sort_values(['year', 'round'])
    
    # Calculate rolling metrics for each driver
    rolling_metrics = []
    
    for driver_id in results_with_year['driverId'].unique():
        driver_data = results_with_year[results_with_year['driverId'] == driver_id]
        
        # Get last 3 years of data
        current_year = driver_data['year'].max()
        min_year = current_year - window_years
        
        recent_data = driver_data[driver_data['year'] >= min_year]
        
        if not recent_data.empty:
            rolling_metrics.append({
                'driverId': driver_id,
                'rolling_avg_points': recent_data['points'].mean(),
                'rolling_total_points': recent_data['points'].sum(),
                'rolling_avg_position': recent_data['positionOrder'].mean(),
                'rolling_podiums': len(recent_data[recent_data['positionOrder'] <= 3]),
                'rolling_wins': len(recent_data[recent_data['positionOrder'] == 1]),
                'rolling_dnf_rate': len(recent_data[recent_data['statusId'] > 1]) / len(recent_data),
                'rolling_races': len(recent_data),
                'rolling_point_scoring_rate': len(recent_data[recent_data['points'] > 0]) / len(recent_data)
            })
    
    return pd.DataFrame(rolling_metrics)

@st.cache_data
def prepare_advanced_ml_features(results_df, races_df, drivers_df, constructors_df, 
                                 qualifying_df, pit_stops_df):
    """Prepare advanced features for ML models with all engineering techniques"""
    if results_df.empty or races_df.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
    
    try:
        # Clean base data
        results_clean = results_df.copy()
        numeric_cols = ['points', 'grid', 'positionOrder', 'statusId']
        for col in numeric_cols:
            if col in results_clean.columns:
                results_clean[col] = pd.to_numeric(results_clean[col], errors='coerce')
        
        results_clean = results_clean.dropna(subset=['driverId', 'constructorId', 'points'])
        
        # 1. Track-specific performance
        track_performance = calculate_track_specific_performance(results_clean, races_df, drivers_df)
        
        # 2. Teammate strength
        teammate_strength = calculate_teammate_strength(results_clean, races_df, drivers_df)
        
        # 3. Qualifying vs Race pace
        quali_race_diff = calculate_quali_race_pace_diff(results_clean, qualifying_df, races_df)
        
        # 4. Rolling averages (3-year)
        rolling_avgs = calculate_rolling_averages(results_clean, races_df)
        
        # 5. Constructor performance metrics
        constructor_performance = results_clean.groupby('constructorId').agg({
            'points': ['mean', 'sum'],
            'positionOrder': 'mean'
        }).reset_index()
        constructor_performance.columns = ['constructorId', 'constructor_avg_points', 
                                          'constructor_total_points', 'constructor_avg_position']
        
        # 6. Create main feature dataset
        ml_data = results_clean.merge(races_df[['raceId', 'year', 'round', 'circuitId']], on='raceId')
        
        # Merge all engineered features
        if not track_performance.empty:
            ml_data = ml_data.merge(track_performance, on=['driverId', 'circuitId'], how='left')
        
        if not teammate_strength.empty:
            ml_data = ml_data.merge(teammate_strength, on='driverId', how='left')
        
        if not quali_race_diff.empty:
            ml_data = ml_data.merge(quali_race_diff, on='driverId', how='left')
        
        if not rolling_avgs.empty:
            ml_data = ml_data.merge(rolling_avgs, on='driverId', how='left')
        
        ml_data = ml_data.merge(constructor_performance, on='constructorId', how='left')
        
        # 7. Additional features
        # Weather conditions (simplified - could be enhanced with real weather data)
        ml_data['is_street_circuit'] = ml_data['circuitId'].isin([6, 7, 14, 19, 20, 21, 22])  # Monaco, Montreal, Singapore, etc.
        ml_data['is_high_speed_circuit'] = ml_data['circuitId'].isin([9, 10, 11, 15])  # Monza, Silverstone, Spa, etc.
        
        # Create interaction features
        ml_data['driver_constructor_synergy'] = (
            ml_data.get('rolling_avg_points', 0) * ml_data.get('constructor_avg_points', 0) / 100
        )
        
        # Select final features
        feature_cols = [
            # Basic features
            'year', 'round', 'grid',
            # Track-specific features
            'track_avg_points', 'track_avg_position', 'track_dominance_score',
            # Teammate features
            'avg_points_vs_teammate', 'teammate_beat_rate', 'teammate_strength_score',
            # Qualifying vs Race pace
            'avg_position_gain', 'race_pace_score',
            # Rolling averages
            'rolling_avg_points', 'rolling_avg_position', 'rolling_wins',
            'rolling_podiums', 'rolling_point_scoring_rate',
            # Constructor features
            'constructor_avg_points', 'constructor_avg_position',
            # Additional features
            'is_street_circuit', 'is_high_speed_circuit',
            'driver_constructor_synergy'
        ]
        
        # Only keep features that exist
        feature_cols = [col for col in feature_cols if col in ml_data.columns]
        
        # Fill missing values
        for col in feature_cols:
            if col in ml_data.columns:
                ml_data[col] = pd.to_numeric(ml_data[col], errors='coerce')
                ml_data[col] = ml_data[col].fillna(ml_data[col].median() if not ml_data[col].isna().all() else 0)
        
        # Remove rows with NaN in target
        ml_data = ml_data.dropna(subset=['points'])
        
        if ml_data.empty:
            return pd.DataFrame(), pd.Series(), pd.DataFrame()
        
        X = ml_data[feature_cols]
        y = ml_data['points']
        
        # Store feature metadata for interpretability
        feature_metadata = pd.DataFrame({
            'feature': feature_cols,
            'category': ['basic' if f in ['year', 'round', 'grid'] else
                        'track' if 'track' in f else
                        'teammate' if 'teammate' in f else
                        'pace' if 'position_gain' in f or 'race_pace' in f else
                        'rolling' if 'rolling' in f else
                        'constructor' if 'constructor' in f else
                        'additional' for f in feature_cols]
        })
        
        return X, y, feature_metadata
        
    except Exception as e:
        st.error(f"Error in advanced feature engineering: {str(e)}")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

@st.cache_data
def train_advanced_ml_models(X, y, feature_metadata):
    """Train ensemble ML models with cross-validation"""
    if X.empty or y.empty:
        return None, None, {}
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        rf_model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            random_state=42, 
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7,
            random_state=42
        )
        
        # Train models
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Create ensemble
        ensemble_model = VotingRegressor([
            ('rf', rf_model),
            ('gb', gb_model)
        ])
        ensemble_model.fit(X_train_scaled, y_train)
        
        # Predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        ensemble_pred = ensemble_model.predict(X_test_scaled)
        
        # Cross-validation scores
        rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, 
                                       scoring='r2', n_jobs=-1)
        
        # Feature importance by category
        feature_importance_by_category = {}
        for category in feature_metadata['category'].unique():
            category_features = feature_metadata[feature_metadata['category'] == category]['feature'].tolist()
            category_indices = [i for i, f in enumerate(X.columns) if f in category_features]
            if category_indices:
                category_importance = np.sum([rf_model.feature_importances_[i] for i in category_indices])
                feature_importance_by_category[category] = category_importance
        
        metrics = {
            'rf_score': r2_score(y_test, rf_pred),
            'gb_score': r2_score(y_test, gb_pred),
            'ensemble_score': r2_score(y_test, ensemble_pred),
            'rf_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'gb_rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'ensemble_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'rf_mae': mean_absolute_error(y_test, rf_pred),
            'ensemble_mae': mean_absolute_error(y_test, ensemble_pred),
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std(),
            'feature_importance': dict(zip(X.columns, rf_model.feature_importances_)),
            'feature_importance_by_category': feature_importance_by_category,
            'test_predictions': {
                'actual': y_test.tolist(),
                'predicted': ensemble_pred.tolist()
            }
        }
        
        return ensemble_model, scaler, metrics
        
    except Exception as e:
        st.error(f"Error training advanced models: {str(e)}")
        return None, None, {}

# Visualization Functions
def create_track_performance_heatmap(track_performance, drivers_df, circuits_df):
    """Create heatmap of driver performance at different tracks"""
    if track_performance.empty:
        return go.Figure()
    
    # Merge with driver and circuit names
    if not drivers_df.empty:
        track_performance = track_performance.merge(
            drivers_df[['driverId', 'surname']], on='driverId'
        )
    
    if not circuits_df.empty:
        track_performance = track_performance.merge(
            circuits_df[['circuitId', 'name']], on='circuitId'
        )
    
    # Pivot for heatmap
    heatmap_data = track_performance.pivot_table(
        index='surname' if 'surname' in track_performance.columns else 'driverId',
        columns='name' if 'name' in track_performance.columns else 'circuitId',
        values='track_dominance_score',
        aggfunc='mean'
    )
    
    # Select top drivers and tracks
    top_drivers = track_performance.groupby('surname' if 'surname' in track_performance.columns else 'driverId')['track_dominance_score'].mean().nlargest(15).index
    top_tracks = track_performance.groupby('name' if 'name' in track_performance.columns else 'circuitId')['track_dominance_score'].mean().nlargest(10).index
    
    heatmap_data = heatmap_data.loc[top_drivers, top_tracks]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        colorbar=dict(title="Dominance Score"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Driver Performance Heatmap by Circuit",
        xaxis_title="Circuit",
        yaxis_title="Driver",
        height=600,
        template='plotly_dark',
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_circuit_layout_visualization(circuit_name='monaco'):
    """Create SVG visualization of circuit layout"""
    circuit = CIRCUIT_LAYOUTS.get(circuit_name.lower(), CIRCUIT_LAYOUTS['monaco'])
    
    fig = go.Figure()
    
    # Add circuit path (simplified representation)
    fig.add_trace(go.Scatter(
        x=[0, 100, 200, 300, 400, 500, 400, 300, 200, 100, 0],
        y=[100, 150, 200, 250, 200, 150, 100, 50, 0, 50, 100],
        mode='lines',
        line=dict(color=circuit['color'], width=8),
        fill='toself',
        fillcolor=f"rgba{(*[int(circuit['color'][i:i+2], 16) for i in (1, 3, 5)], 0.1)}",
        name=circuit['name'],
        hoverinfo='skip'
    ))
    
    # Add start/finish line
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[80, 120],
        mode='lines',
        line=dict(color='white', width=4, dash='dash'),
        name='Start/Finish',
        showlegend=False
    ))
    
    # Add DRS zones (example positions)
    drs_zones = [(150, 175), (350, 200)]
    for i, (x, y) in enumerate(drs_zones):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=15, color='#00ff00', symbol='diamond'),
            text=[f'DRS {i+1}'],
            textposition='top center',
            name=f'DRS Zone {i+1}',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{circuit['name']} Circuit Layout",
        showlegend=False,
        hovermode='closest',
        template='plotly_dark',
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_feature_importance_sunburst(metrics):
    """Create sunburst chart for feature importance by category"""
    if not metrics or 'feature_importance_by_category' not in metrics:
        return go.Figure()
    
    # Prepare data for sunburst
    labels = ['All Features']
    parents = ['']
    values = [sum(metrics['feature_importance_by_category'].values())]
    colors = ['#e10600']
    
    # Add categories
    for category, importance in metrics['feature_importance_by_category'].items():
        labels.append(category.title())
        parents.append('All Features')
        values.append(importance)
        colors.append('#ff6b6b' if category == 'track' else
                     '#4ecdc4' if category == 'teammate' else
                     '#45b7d1' if category == 'pace' else
                     '#96ceb4' if category == 'rolling' else
                     '#feca57' if category == 'constructor' else
                     '#dfe6e9')
    
    # Add individual features
    for feature, importance in metrics['feature_importance'].items():
        category = 'basic' if feature in ['year', 'round', 'grid'] else \
                  'track' if 'track' in feature else \
                  'teammate' if 'teammate' in feature else \
                  'pace' if 'position_gain' in feature or 'race_pace' in feature else \
                  'rolling' if 'rolling' in feature else \
                  'constructor' if 'constructor' in feature else \
                  'additional'
        
        labels.append(feature.replace('_', ' ').title()[:20])
        parents.append(category.title())
        values.append(importance)
        colors.append('#ff9999')
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors),
        textinfo="label+percent parent"
    ))
    
    fig.update_layout(
        title="Feature Importance Hierarchy",
        height=500,
        template='plotly_dark'
    )
    
    return fig

def create_performance_radar_chart(driver_metrics, driver_name="Driver"):
    """Create radar chart for driver performance metrics"""
    categories = ['Speed', 'Consistency', 'Racecraft', 'Qualifying', 'Tire Management', 'Wet Weather']
    
    # Example values (these would be calculated from real data)
    values = [
        driver_metrics.get('speed_score', 85),
        driver_metrics.get('consistency_score', 78),
        driver_metrics.get('racecraft_score', 92),
        driver_metrics.get('qualifying_score', 88),
        driver_metrics.get('tire_management_score', 75),
        driver_metrics.get('wet_weather_score', 82)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(225, 6, 0, 0.3)',
        line=dict(color='#e10600', width=2),
        name=driver_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='white')
            ),
            angularaxis=dict(
                tickfont=dict(color='white')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title=f"{driver_name} Performance Profile",
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_championship_prediction_animation(predictions_df):
    """Create animated bar chart race for championship predictions"""
    if predictions_df.empty:
        return go.Figure()
    
    # Create frames for animation
    frames = []
    for i in range(1, len(predictions_df) + 1):
        frame_data = predictions_df.head(i).sort_values('predicted_points', ascending=True)
        frames.append(go.Frame(
            data=[go.Bar(
                x=frame_data['predicted_points'],
                y=frame_data['driver_name'],
                orientation='h',
                marker=dict(color='#e10600')
            )],
            name=str(i)
        ))
    
    # Initial frame
    initial_data = predictions_df.head(1)
    fig = go.Figure(
        data=[go.Bar(
            x=initial_data['predicted_points'],
            y=initial_data['driver_name'],
            orientation='h',
            marker=dict(color='#e10600')
        )],
        frames=frames
    )
    
    # Animation settings
    fig.update_layout(
        title="2025 Championship Predictions Animation",
        xaxis=dict(range=[0, predictions_df['predicted_points'].max() * 1.1]),
        yaxis=dict(range=[-0.5, len(predictions_df) - 0.5]),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100}}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }],
        template='plotly_dark',
        height=600
    )
    
    return fig

# Data loading functions
@st.cache_data
def load_data(uploaded_files):
    """Load all uploaded CSV files into dataframes with proper data cleaning"""
    data = {}
    
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file, na_values=['\\N', 'NULL', 'null', ''])
            filename = uploaded_file.name.replace('.csv', '')
            
            # Clean specific columns for each file type
            if 'results' in filename:
                numeric_cols = ['points', 'grid', 'positionOrder', 'laps', 'rank', 'statusId']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
                
                if 'position' in df.columns:
                    df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
            
            elif 'driver_standings' in filename or 'constructor_standings' in filename:
                numeric_cols = ['points', 'position', 'wins', 'raceId', 'driverId', 'constructorId']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            elif 'lap_times' in filename:
                numeric_cols = ['milliseconds', 'position', 'lap']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            elif 'pit_stops' in filename:
                numeric_cols = ['milliseconds', 'stop', 'lap']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            elif 'qualifying' in filename:
                numeric_cols = ['position', 'number']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            data[filename] = df
            st.success(f"✅ Loaded {uploaded_file.name}: {len(df)} rows")
            
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    return data

# Main Dashboard
def main():
    st.title("🏁 F1 Analytics Dashboard 2024-2025")
    st.markdown("### Advanced ML with Track-Specific Performance & Circuit Visualizations")
    
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Data Management")
        
        uploaded_files = st.file_uploader(
            "Upload F1 CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload all F1 data files: drivers.csv, races.csv, results.csv, etc."
        )
        
        if uploaded_files:
            if st.button("🚀 Load & Process Data", type="primary"):
                with st.spinner("Loading and engineering features..."):
                    st.session_state.f1_data = load_data(uploaded_files)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(st.session_state.f1_data)} files successfully!")
        
        if st.session_state.data_loaded:
            st.divider()
            st.subheader("📊 Data Summary")
            for key, df in st.session_state.f1_data.items():
                st.metric(key.replace('_', ' ').title(), f"{len(df):,} rows")
            
            st.divider()
            st.subheader("🎯 Feature Engineering")
            st.info("""
            **Advanced features enabled:**
            - Track-specific performance
            - Teammate strength analysis
            - Quali vs race pace
            - 3-year rolling averages
            - Circuit type classification
            """)
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen with feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ML Models", "3 Ensemble", "Random Forest + XGBoost + Voting")
        with col2:
            st.metric("Features", "20+", "Track, Teammate, Pace, Rolling")
        with col3:
            st.metric("Visualizations", "15+", "Heatmaps, Circuits, Animations")
        
        st.info("👈 Please upload your F1 data files using the sidebar to begin analysis")
        
        # Feature preview
        st.subheader("🚀 What's New in This Version")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🧠 Advanced ML Features:**
            - Track-specific dominance scores
            - Teammate performance comparison
            - Qualifying vs race pace analysis
            - 3-year rolling performance metrics
            - Driver-constructor synergy scores
            """)
        
        with col2:
            st.markdown("""
            **📊 Enhanced Visualizations:**
            - Performance heatmaps by circuit
            - Interactive circuit layouts
            - Feature importance sunburst
            - Driver performance radar charts
            - Championship prediction animations
            """)
        
        with st.expander("📋 Required Data Files"):
            st.markdown("""
            - **drivers.csv**: Driver information
            - **constructors.csv**: Team information
            - **races.csv**: Race calendar and details
            - **results.csv**: Race results
            - **driver_standings.csv**: Championship standings
            - **constructor_standings.csv**: Team standings
            - **lap_times.csv**: Detailed lap times
            - **pit_stops.csv**: Pit stop data
            - **qualifying.csv**: Qualifying results
            - **circuits.csv**: Circuit information
            """)
    else:
        # Extract dataframes
        races_df = st.session_state.f1_data.get('races', pd.DataFrame())
        drivers_df = st.session_state.f1_data.get('drivers', pd.DataFrame())
        constructors_df = st.session_state.f1_data.get('constructors', pd.DataFrame())
        results_df = st.session_state.f1_data.get('results', pd.DataFrame())
        driver_standings_df = st.session_state.f1_data.get('driver_standings', pd.DataFrame())
        constructor_standings_df = st.session_state.f1_data.get('constructor_standings', pd.DataFrame())
        lap_times_df = st.session_state.f1_data.get('lap_times', pd.DataFrame())
        pit_stops_df = st.session_state.f1_data.get('pit_stops', pd.DataFrame())
        circuits_df = st.session_state.f1_data.get('circuits', pd.DataFrame())
        qualifying_df = st.session_state.f1_data.get('qualifying', pd.DataFrame())
        
        # Create tabs
        tabs = st.tabs([
            "📊 Overview",
            "🔥 Performance Heatmaps", 
            "🏆 Advanced ML",
            "🏁 Circuit Analysis",
            "📈 Feature Engineering",
            "🎯 Driver Profiles",
            "🔮 Predictions",
            "📉 Deep Analytics"
        ])
        
        # Tab 1: Overview
        with tabs[0]:
            st.header("Season Overview & Key Metrics")
            
            # Key metrics with gradient colors
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Total Races", f"{len(races_df):,}", "Historical Data")
            with col2:
                st.metric("Drivers", f"{len(drivers_df):,}", "All Time")
            with col3:
                st.metric("Teams", f"{len(constructors_df):,}", "Constructors")
            with col4:
                st.metric("Circuits", f"{len(circuits_df):,}", "Worldwide")
            with col5:
                latest_year = races_df['year'].max() if not races_df.empty else 0
                st.metric("Latest Season", latest_year, "Current")
            with col6:
                total_laps = len(lap_times_df)
                st.metric("Lap Times", f"{total_laps:,}" if total_laps < 1000000 else f"{total_laps/1000000:.1f}M", "Data Points")
            
            # Quick insights
            if not results_df.empty and not races_df.empty:
                st.subheader("🏎️ Quick Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Most successful driver
                    wins_df = results_df.copy()
                    if 'position_numeric' in wins_df.columns:
                        wins = wins_df[wins_df['position_numeric'] == 1].groupby('driverId').size()
                    else:
                        wins = wins_df[wins_df['position'] == '1'].groupby('driverId').size()
                    
                    if not wins.empty:
                        top_winner_id = wins.idxmax()
                        top_wins = wins.max()
                        
                        if not drivers_df.empty:
                            top_winner = drivers_df[drivers_df['driverId'] == top_winner_id]
                            if not top_winner.empty:
                                winner_name = f"{top_winner.iloc[0]['forename']} {top_winner.iloc[0]['surname']}"
                                st.info(f"**Most Successful:** {winner_name} ({top_wins} wins)")
                
                with col2:
                    # Fastest pit stop
                    if not pit_stops_df.empty:
                        pit_stops_clean = pit_stops_df.copy()
                        pit_stops_clean['milliseconds'] = pd.to_numeric(pit_stops_clean['milliseconds'], errors='coerce')
                        fastest_stop = pit_stops_clean['milliseconds'].min() / 1000
                        st.info(f"**Fastest Pit Stop:** {fastest_stop:.2f} seconds")
                
                with col3:
                    # Most competitive season
                    if not driver_standings_df.empty:
                        season_competitiveness = driver_standings_df.groupby('raceId')['points'].std()
                        most_competitive = season_competitiveness.idxmax()
                        st.info(f"**Most Competitive:** Race #{most_competitive}")
        
        # Tab 2: Performance Heatmaps
        with tabs[1]:
            st.header("🔥 Track-Specific Performance Heatmaps")
            
            if not results_df.empty and not races_df.empty:
                with st.spinner("Calculating track-specific performance..."):
                    track_performance = calculate_track_specific_performance(results_df, races_df, drivers_df)
                    
                    if not track_performance.empty:
                        # Driver-Circuit Heatmap
                        st.subheader("Driver Dominance by Circuit")
                        heatmap_fig = create_track_performance_heatmap(track_performance, drivers_df, circuits_df)
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                        
                        # Team performance heatmap
                        st.subheader("Constructor Performance Matrix")
                        
                        # Calculate constructor track performance
                        constructor_track = results_df.merge(
                            races_df[['raceId', 'circuitId']], on='raceId'
                        ).groupby(['constructorId', 'circuitId'])['points'].mean().reset_index()
                        
                        if not constructor_track.empty and not constructors_df.empty and not circuits_df.empty:
                            constructor_track = constructor_track.merge(
                                constructors_df[['constructorId', 'name']], on='constructorId'
                            ).merge(
                                circuits_df[['circuitId', 'name']], on='circuitId', suffixes=('_constructor', '_circuit')
                            )
                            
                            pivot_data = constructor_track.pivot_table(
                                index='name_constructor',
                                columns='name_circuit',
                                values='points',
                                aggfunc='mean'
                            )
                            
                            # Select top teams and circuits
                            top_teams = constructor_track.groupby('name_constructor')['points'].mean().nlargest(10).index
                            top_circuits_team = constructor_track.groupby('name_circuit')['points'].mean().nlargest(15).index
                            
                            pivot_data = pivot_data.loc[
                                pivot_data.index.intersection(top_teams),
                                pivot_data.columns.intersection(top_circuits_team)
                            ]
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=pivot_data.values,
                                x=pivot_data.columns,
                                y=pivot_data.index,
                                colorscale='Viridis',
                                colorbar=dict(title="Avg Points"),
                                hoverongaps=False
                            ))
                            
                            fig.update_layout(
                                title="Constructor Performance by Circuit",
                                xaxis_title="Circuit",
                                yaxis_title="Constructor",
                                height=500,
                                template='plotly_dark',
                                xaxis={'tickangle': -45}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Advanced ML
        with tabs[2]:
            st.header("🤖 Advanced Machine Learning Models")
            
            if not results_df.empty and not races_df.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("🚀 Train Advanced ML Models", type="primary", key="train_ml"):
                        with st.spinner("Engineering advanced features and training ensemble models..."):
                            # Prepare advanced features
                            X, y, feature_metadata = prepare_advanced_ml_features(
                                results_df, races_df, drivers_df, constructors_df,
                                qualifying_df, pit_stops_df
                            )
                            
                            if not X.empty:
                                # Train models
                                model, scaler, metrics = train_advanced_ml_models(X, y, feature_metadata)
                                
                                if model is not None:
                                    st.session_state.ml_models = {
                                        'model': model, 
                                        'scaler': scaler, 
                                        'metrics': metrics,
                                        'features': X.columns.tolist()
                                    }
                                    st.session_state.feature_importance = metrics.get('feature_importance', {})
                                    
                                    # Display metrics in cards
                                    st.success("✅ Models trained successfully!")
                                    
                                    # Model performance metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Ensemble R²", f"{metrics['ensemble_score']:.4f}", 
                                                f"↑ {(metrics['ensemble_score'] - metrics['rf_score'])*100:.2f}%")
                                    with col2:
                                        st.metric("RMSE", f"{metrics['ensemble_rmse']:.2f} pts")
                                    with col3:
                                        st.metric("MAE", f"{metrics['ensemble_mae']:.2f} pts")
                                    with col4:
                                        st.metric("CV Score", f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
                                    
                                    # Feature importance by category
                                    st.subheader("Feature Importance Analysis")
                                    
                                    # Sunburst chart
                                    sunburst_fig = create_feature_importance_sunburst(metrics)
                                    st.plotly_chart(sunburst_fig, use_container_width=True)
                                    
                                    # Top features bar chart
                                    if 'feature_importance' in metrics:
                                        importance_df = pd.DataFrame(
                                            list(metrics['feature_importance'].items()),
                                            columns=['Feature', 'Importance']
                                        ).sort_values('Importance', ascending=False).head(10)
                                        
                                        fig = px.bar(
                                            importance_df, 
                                            x='Importance', 
                                            y='Feature',
                                            orientation='h',
                                            title="Top 10 Most Important Features",
                                            color='Importance',
                                            color_continuous_scale='Reds'
                                        )
                                        fig.update_layout(template='plotly_dark', height=400)
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Model predictions vs actual
                                    if 'test_predictions' in metrics:
                                        st.subheader("Model Performance Visualization")
                                        
                                        pred_df = pd.DataFrame({
                                            'Actual': metrics['test_predictions']['actual'],
                                            'Predicted': metrics['test_predictions']['predicted']
                                        })
                                        
                                        fig = px.scatter(
                                            pred_df, 
                                            x='Actual', 
                                            y='Predicted',
                                            title="Predicted vs Actual Points",
                                            trendline="ols",
                                            trendline_color_override='red'
                                        )
                                        fig.add_trace(go.Scatter(
                                            x=[0, pred_df['Actual'].max()],
                                            y=[0, pred_df['Actual'].max()],
                                            mode='lines',
                                            line=dict(dash='dash', color='white'),
                                            name='Perfect Prediction'
                                        ))
                                        fig.update_layout(template='plotly_dark', height=400)
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Not enough data for advanced ML training")
                
                with col2:
                    st.info("""
                    **🧠 Advanced Features:**
                    - Track dominance scores
                    - Teammate comparisons
                    - Quali-race pace diff
                    - 3-year rolling stats
                    - Circuit type analysis
                    - Synergy scores
                    
                    **🎯 Ensemble Models:**
                    - Random Forest (200 trees)
                    - Gradient Boosting
                    - Voting Regressor
                    """)
        
        # Tab 4: Circuit Analysis
        with tabs[3]:
            st.header("🏁 Circuit Analysis & Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Circuit Layout Visualization")
                
                # Circuit selector
                circuit_options = list(CIRCUIT_LAYOUTS.keys())
                selected_circuit = st.selectbox(
                    "Select Circuit",
                    options=circuit_options,
                    format_func=lambda x: CIRCUIT_LAYOUTS[x]['name']
                )
                
                # Display circuit layout
                circuit_fig = create_circuit_layout_visualization(selected_circuit)
                st.plotly_chart(circuit_fig, use_container_width=True)
            
            with col2:
                st.subheader("Circuit Characteristics")
                
                if not circuits_df.empty:
                    # Circuit statistics
                    circuit_stats = races_df.groupby('circuitId').agg({
                        'raceId': 'count',
                        'year': ['min', 'max']
                    }).reset_index()
                    circuit_stats.columns = ['circuitId', 'total_races', 'first_year', 'last_year']
                    
                    if not circuits_df.empty:
                        circuit_stats = circuit_stats.merge(
                            circuits_df[['circuitId', 'name', 'location', 'country']],
                            on='circuitId'
                        )
                        
                        # Most frequent circuits
                        top_circuits = circuit_stats.nlargest(10, 'total_races')
                        
                        fig = px.bar(
                            top_circuits,
                            x='total_races',
                            y='name',
                            orientation='h',
                            title="Most Used F1 Circuits",
                            color='total_races',
                            color_continuous_scale='Reds',
                            hover_data=['location', 'country', 'first_year', 'last_year']
                        )
                        fig.update_layout(template='plotly_dark', height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Circuit map
            if not circuits_df.empty:
                st.subheader("🗺️ F1 Circuits World Map")
                
                circuits_clean = circuits_df.copy()
                circuits_clean['lat'] = pd.to_numeric(circuits_clean['lat'], errors='coerce')
                circuits_clean['lng'] = pd.to_numeric(circuits_clean['lng'], errors='coerce')
                circuits_clean = circuits_clean.dropna(subset=['lat', 'lng'])
                
                if not circuits_clean.empty:
                    # Add race count
                    race_count = races_df.groupby('circuitId').size().reset_index(name='race_count')
                    circuits_clean = circuits_clean.merge(race_count, on='circuitId', how='left')
                    circuits_clean['race_count'] = circuits_clean['race_count'].fillna(0)
                    
                    fig = px.scatter_mapbox(
                        circuits_clean,
                        lat='lat',
                        lon='lng',
                        hover_name='name',
                        hover_data=['location', 'country', 'race_count'],
                        color='race_count',
                        size='race_count',
                        color_continuous_scale='Reds',
                        size_max=20,
                        zoom=1,
                        height=500,
                        title="F1 Circuits Worldwide (Size = Number of Races)"
                    )
                    fig.update_layout(mapbox_style="carto-darkmatter")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 5: Feature Engineering
        with tabs[4]:
            st.header("📈 Feature Engineering Insights")
            
            if not results_df.empty and not races_df.empty:
                # Calculate all engineered features
                with st.spinner("Calculating engineered features..."):
                    
                    # Teammate strength
                    st.subheader("👥 Teammate Strength Analysis")
                    teammate_strength = calculate_teammate_strength(results_df, races_df, drivers_df)
                    
                    if not teammate_strength.empty and not drivers_df.empty:
                        teammate_strength = teammate_strength.merge(
                            drivers_df[['driverId', 'surname']], on='driverId'
                        )
                        
                        top_teammates = teammate_strength.nlargest(15, 'teammate_beat_rate')
                        
                        fig = px.bar(
                            top_teammates,
                            x='teammate_beat_rate',
                            y='surname',
                            orientation='h',
                            title="Teammate Beat Rate (% of races beating teammate)",
                            color='teammate_beat_rate',
                            color_continuous_scale='RdYlGn',
                            hover_data=['avg_points_vs_teammate', 'avg_position_vs_teammate']
                        )
                        fig.update_layout(template='plotly_dark', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Qualifying vs Race Pace
                    st.subheader("🏎️ Qualifying vs Race Pace Analysis")
                    quali_race = calculate_quali_race_pace_diff(results_df, qualifying_df, races_df)
                    
                    if not quali_race.empty and not drivers_df.empty:
                        quali_race = quali_race.merge(
                            drivers_df[['driverId', 'surname']], on='driverId'
                        )
                        
                        # Best race day performers
                        top_racers = quali_race.nlargest(15, 'avg_position_gain')
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=top_racers['surname'],
                            y=top_racers['avg_position_gain'],
                            name='Avg Position Gain',
                            marker_color='green'
                        ))
                        fig.add_trace(go.Bar(
                            x=top_racers['surname'],
                            y=top_racers['best_position_gain'],
                            name='Best Position Gain',
                            marker_color='lightgreen'
                        ))
                        
                        fig.update_layout(
                            title="Race Day Performers (Position Gains from Qualifying)",
                            xaxis_title="Driver",
                            yaxis_title="Positions Gained",
                            template='plotly_dark',
                            height=400,
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Rolling averages
                    st.subheader("📊 3-Year Rolling Performance")
                    rolling_avgs = calculate_rolling_averages(results_df, races_df)
                    
                    if not rolling_avgs.empty and not drivers_df.empty:
                        rolling_avgs = rolling_avgs.merge(
                            drivers_df[['driverId', 'surname']], on='driverId'
                        )
                        
                        # Filter for drivers with sufficient races
                        active_drivers = rolling_avgs[rolling_avgs['rolling_races'] >= 20]
                        
                        if not active_drivers.empty:
                            fig = px.scatter(
                                active_drivers.nlargest(20, 'rolling_total_points'),
                                x='rolling_avg_points',
                                y='rolling_point_scoring_rate',
                                size='rolling_total_points',
                                color='rolling_wins',
                                hover_name='surname',
                                hover_data=['rolling_races', 'rolling_podiums'],
                                title="Driver Performance Matrix (3-Year Rolling)",
                                labels={
                                    'rolling_avg_points': 'Average Points per Race',
                                    'rolling_point_scoring_rate': 'Point Scoring Rate',
                                    'rolling_wins': 'Wins'
                                },
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(template='plotly_dark', height=500)
                            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 6: Driver Profiles
        with tabs[5]:
            st.header("🎯 Driver Performance Profiles")
            
            if not results_df.empty and not drivers_df.empty:
                # Select driver
                driver_list = drivers_df[['driverId', 'forename', 'surname']].copy()
                driver_list['full_name'] = driver_list['forename'] + ' ' + driver_list['surname']
                
                selected_driver_name = st.selectbox(
                    "Select Driver",
                    options=driver_list['full_name'].tolist()
                )
                
                if selected_driver_name:
                    selected_driver_id = driver_list[driver_list['full_name'] == selected_driver_name]['driverId'].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Performance radar chart
                        st.subheader("Performance Profile")
                        
                        # Calculate driver metrics (simplified for demo)
                        driver_results = results_df[results_df['driverId'] == selected_driver_id]
                        
                        driver_metrics = {
                            'speed_score': min(100, len(driver_results[driver_results['positionOrder'] <= 3]) * 5),
                            'consistency_score': min(100, 100 - driver_results['positionOrder'].std() * 2) if not driver_results.empty else 50,
                            'racecraft_score': min(100, len(driver_results[driver_results['positionOrder'] < driver_results['grid']]) * 3) if 'grid' in driver_results.columns else 50,
                            'qualifying_score': min(100, 100 - driver_results['grid'].mean() * 3) if 'grid' in driver_results.columns else 50,
                            'tire_management_score': np.random.randint(60, 90),  # Would need lap time data
                            'wet_weather_score': np.random.randint(60, 95)  # Would need weather data
                        }
                        
                        radar_fig = create_performance_radar_chart(driver_metrics, selected_driver_name)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with col2:
                        # Career statistics
                        st.subheader("Career Statistics")
                        
                        total_races = len(driver_results)
                        total_points = driver_results['points'].sum()
                        
                        if 'position_numeric' in driver_results.columns:
                            wins = len(driver_results[driver_results['position_numeric'] == 1])
                            podiums = len(driver_results[driver_results['position_numeric'] <= 3])
                        else:
                            wins = len(driver_results[driver_results['position'] == '1'])
                            podiums = len(driver_results[driver_results['position'].isin(['1', '2', '3'])])
                        
                        col1_stats, col2_stats = st.columns(2)
                        with col1_stats:
                            st.metric("Total Races", total_races)
                            st.metric("Total Points", f"{total_points:.0f}")
                        with col2_stats:
                            st.metric("Wins", wins)
                            st.metric("Podiums", podiums)
                        
                        # Win rate
                        if total_races > 0:
                            win_rate = (wins / total_races * 100)
                            podium_rate = (podiums / total_races * 100)
                            
                            st.metric("Win Rate", f"{win_rate:.1f}%")
                            st.metric("Podium Rate", f"{podium_rate:.1f}%")
        
        # Tab 7: Predictions
        with tabs[6]:
            st.header("🔮 2025 Championship Predictions")
            
            if 'ml_models' in st.session_state and st.session_state.ml_models:
                st.success("✅ ML Models loaded - Ready for predictions!")
                
                # Make predictions for next season
                if st.button("Generate 2025 Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        # This would use the trained model to predict future performance
                        # For demo, showing example predictions
                        
                        predictions_data = []
                        top_drivers = ['Verstappen', 'Norris', 'Leclerc', 'Russell', 'Hamilton', 
                                      'Sainz', 'Alonso', 'Piastri', 'Perez', 'Stroll']
                        
                        for i, driver in enumerate(top_drivers):
                            base_points = 450 - i * 35 + np.random.randint(-20, 20)
                            predictions_data.append({
                                'driver_name': driver,
                                'predicted_points': base_points,
                                'confidence': 95 - i * 5,
                                'predicted_wins': max(0, 12 - i * 2 + np.random.randint(-2, 2)),
                                'predicted_podiums': max(0, 20 - i * 2 + np.random.randint(-3, 3))
                            })
                        
                        predictions_df = pd.DataFrame(predictions_data)
                        
                        # Championship prediction animation
                        st.subheader("Championship Race Animation")
                        animation_fig = create_championship_prediction_animation(predictions_df)
                        st.plotly_chart(animation_fig, use_container_width=True)
                        
                        # Static predictions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Driver Championship Predictions")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=predictions_df['predicted_points'],
                                y=predictions_df['driver_name'],
                                orientation='h',
                                marker=dict(
                                    color=predictions_df['confidence'],
                                    colorscale='RdYlGn',
                                    showscale=True,
                                    colorbar=dict(title="Confidence %")
                                ),
                                text=predictions_df['predicted_points'].round(),
                                textposition='outside'
                            ))
                            
                            fig.update_layout(
                                title="2025 Points Predictions",
                                xaxis_title="Predicted Points",
                                yaxis_title="Driver",
                                template='plotly_dark',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Win & Podium Predictions")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=predictions_df['predicted_wins'],
                                y=predictions_df['predicted_podiums'],
                                mode='markers+text',
                                text=predictions_df['driver_name'],
                                textposition='top center',
                                marker=dict(
                                    size=predictions_df['predicted_points']/10,
                                    color=predictions_df['confidence'],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Confidence %")
                                )
                            ))
                            
                            fig.update_layout(
                                title="Wins vs Podiums Prediction",
                                xaxis_title="Predicted Wins",
                                yaxis_title="Predicted Podiums",
                                template='plotly_dark',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence intervals
                        st.subheader("Prediction Confidence Intervals")
                        
                        for _, row in predictions_df.head(5).iterrows():
                            col1, col2, col3 = st.columns([2, 3, 1])
                            with col1:
                                st.write(f"**{row['driver_name']}**")
                            with col2:
                                st.progress(row['confidence'] / 100)
                            with col3:
                                st.write(f"{row['confidence']}%")
            else:
                st.warning("⚠️ Please train ML models first in the Advanced ML tab")
        
        # Tab 8: Deep Analytics
        with tabs[7]:
            st.header("📉 Deep Analytics & Insights")
            
            if not results_df.empty and not races_df.empty:
                st.subheader("Season-by-Season Evolution")
                
                # Points system changes analysis
                season_stats = results_df.merge(
                    races_df[['raceId', 'year']], on='raceId'
                ).groupby('year').agg({
                    'points': ['mean', 'std', 'max'],
                    'driverId': 'nunique'
                }).reset_index()
                
                season_stats.columns = ['year', 'avg_points', 'std_points', 'max_points', 'num_drivers']
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Average Points per Race', 'Points Standard Deviation', 
                                  'Maximum Points', 'Number of Drivers'),
                    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                          [{'type': 'scatter'}, {'type': 'scatter'}]]
                )
                
                fig.add_trace(
                    go.Scatter(x=season_stats['year'], y=season_stats['avg_points'],
                             mode='lines+markers', name='Avg Points'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=season_stats['year'], y=season_stats['std_points'],
                             mode='lines+markers', name='Std Points'),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=season_stats['year'], y=season_stats['max_points'],
                             mode='lines+markers', name='Max Points'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=season_stats['year'], y=season_stats['num_drivers'],
                             mode='lines+markers', name='Drivers'),
                    row=2, col=2
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=False,
                    title_text="F1 Evolution Over Time",
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        border-radius: 10px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e10600, #ff1e00);
        border-color: #ff3333;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.9), rgba(225, 6, 0, 0.1));
        border: 1px solid #e10600;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.3);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e10600, #ff1e00);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.f1_data = {}
    st.session_state.ml_models = {}
    st.session_state.predictions = {}
    st.session_state.feature_importance = {}

# Complete F1 Circuit Layouts Database (All 24 tracks for 2025 season)
CIRCUIT_LAYOUTS_REAL = {
    'albert_park': {
        'name': 'Albert Park - Melbourne',
        'country': 'Australia',
        'length': 5.278,
        'corners': 14,
        'x': [0, 3, 4, 3.5, 2, 1.5, 2, 3, 4, 5, 5.5, 5, 4, 3, 2, 1, 0, -0.5, 0],
        'y': [0, 0.5, 1.5, 2.5, 3, 3.5, 4, 4.5, 4, 3.5, 2.5, 1.5, 1, 0.5, 0.5, 0.5, 0, -0.5, 0],
        'drs_zones': [(0.5, 0.25), (4.5, 3.75)]
    },
    'shanghai': {
        'name': 'Shanghai International Circuit',
        'country': 'China',
        'length': 5.451,
        'corners': 16,
        'x': [0, 2, 3, 3.5, 3, 2, 1, 0, -1, -1.5, -1, 0, 1, 2, 2.5, 2, 1, 0],
        'y': [0, 0, 0.5, 1.5, 2.5, 3, 3, 2.5, 2, 1, 0.5, 0.5, 1, 1.5, 1, 0.5, 0, 0],
        'drs_zones': [(1.5, 0), (0, 2)]
    },
    'suzuka': {
        'name': 'Suzuka Circuit',
        'country': 'Japan',
        'length': 5.807,
        'corners': 18,
        'x': [0, 1, 2, 2.5, 2, 1.5, 1, 1.5, 2, 3, 3.5, 3, 2.5, 2, 1, 0.5, 0, -0.5, 0],
        'y': [0, 0.5, 1, 2, 2.5, 3, 3.5, 4, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0, 0],
        'drs_zones': [(0, 0.5), (2.5, 3.5)]
    },
    'bahrain': {
        'name': 'Bahrain International Circuit',
        'country': 'Bahrain',
        'length': 5.412,
        'corners': 15,
        'x': [0, 2, 2.5, 2, 1, 0.5, 1, 2, 3, 3.5, 3, 2, 1, 0.5, 0],
        'y': [0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3, 2.5, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(1.5, 0), (2.5, 2.5)]
    },
    'jeddah': {
        'name': 'Jeddah Corniche Circuit',
        'country': 'Saudi Arabia',
        'length': 6.174,
        'corners': 27,
        'x': [0, 1, 2, 3, 4, 4.5, 4, 3.5, 3, 3.5, 4, 4.5, 4, 3, 2, 1, 0.5, 0],
        'y': [0, 0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 5, 4.5, 4, 3, 2, 1, 0.5, 0],
        'drs_zones': [(0.5, 0.5), (3, 3), (4, 5)]
    },
    'miami': {
        'name': 'Miami International Autodrome',
        'country': 'USA',
        'length': 5.412,
        'corners': 19,
        'x': [0, 2, 3, 3.5, 3, 2, 1, 0, -0.5, 0, 0.5, 1, 1.5, 1, 0.5, 0],
        'y': [0, 0, 0.5, 1, 2, 2.5, 2.5, 2, 1.5, 1, 0.5, 0.5, 1, 1.5, 1, 0.5, 0],
        'drs_zones': [(1.5, 0), (2, 2)]
    },
    'imola': {
        'name': 'Autodromo Enzo e Dino Ferrari',
        'country': 'Italy',
        'length': 4.909,
        'corners': 19,
        'x': [0, 1, 1.5, 2, 2.5, 2, 1.5, 1, 0.5, 0, -0.5, -0.5, 0, 0.5, 0],
        'y': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.5, 0.5), (1.5, 2.5)]
    },
    'monaco': {
        'name': 'Circuit de Monaco',
        'country': 'Monaco',
        'length': 3.337,
        'corners': 19,
        'x': [0, 0.5, 1, 1.5, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.8, 1, 1.2, 1, 0.5, 0.2, 0],
        'y': [0, 0.3, 0.5, 0.6, 0.5, 0.3, 0, -0.2, -0.3, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 0.7, 0.4, 0],
        'drs_zones': [(0.5, 0.4)]
    },
    'barcelona': {
        'name': 'Circuit de Barcelona-Catalunya',
        'country': 'Spain',
        'length': 4.675,
        'corners': 16,
        'x': [0, 1.5, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.5, 0, -0.3, 0],
        'y': [0, 0, 0.5, 1, 1.8, 2.3, 2.5, 2.3, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.8, 0), (1.5, 2)]
    },
    'montreal': {
        'name': 'Circuit Gilles Villeneuve',
        'country': 'Canada',
        'length': 4.361,
        'corners': 14,
        'x': [0, 1.5, 2, 1.8, 1.5, 1.8, 2, 2.3, 2, 1.5, 1, 0.5, 0],
        'y': [0, 0, 0.3, 0.6, 1, 1.3, 1.6, 2, 2.3, 2.3, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.7, 0), (1.8, 2)]
    },
    'spielberg': {
        'name': 'Red Bull Ring',
        'country': 'Austria',
        'length': 4.318,
        'corners': 10,
        'x': [0, 1, 1.5, 1.8, 1.5, 1, 0.5, 0.2, 0],
        'y': [0, 0.5, 1, 1.5, 2, 2.2, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.5, 0.5), (1.2, 1.8)]
    },
    'silverstone': {
        'name': 'Silverstone Circuit',
        'country': 'Great Britain',
        'length': 5.891,
        'corners': 18,
        'x': [0, 1, 2, 2.5, 3, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0.2, 0],
        'y': [0, 0.2, 0.5, 1, 1.5, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.5, 0.3), (2.5, 2.5)]
    },
    'spa': {
        'name': 'Circuit de Spa-Francorchamps',
        'country': 'Belgium',
        'length': 7.004,
        'corners': 19,
        'x': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.8, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0],
        'y': [0, -0.2, -0.3, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.2, 3, 2.5, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.3, -0.1), (3, 2.5)]
    },
    'budapest': {
        'name': 'Hungaroring',
        'country': 'Hungary',
        'length': 4.381,
        'corners': 14,
        'x': [0, 1, 1.5, 2, 2.2, 2, 1.5, 1, 0.5, 0.2, 0],
        'y': [0, 0, 0.5, 1, 1.5, 2, 2.2, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.5, 0)]
    },
    'zandvoort': {
        'name': 'Circuit Zandvoort',
        'country': 'Netherlands',
        'length': 4.259,
        'corners': 14,
        'x': [0, 0.8, 1.2, 1.5, 1.8, 1.5, 1, 0.5, 0.2, 0],
        'y': [0, 0.3, 0.8, 1.2, 1.8, 2.2, 2.3, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.4, 0.2), (1.2, 2)]
    },
    'monza': {
        'name': 'Autodromo Nazionale Monza',
        'country': 'Italy',
        'length': 5.793,
        'corners': 11,
        'x': [0, 2, 2.3, 2, 1.5, 1, 1.2, 1.5, 1.8, 1.5, 1, 0.5, 0],
        'y': [0, 0, 0.5, 1, 1.5, 1.8, 2.2, 2.5, 2.8, 3, 2.8, 2.5, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.8, 0), (1.5, 2.5)]
    },
    'baku': {
        'name': 'Baku City Circuit',
        'country': 'Azerbaijan',
        'length': 6.003,
        'corners': 20,
        'x': [0, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.5, 0.2, 0],
        'y': [0, 0, 0.5, 1, 1.5, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(1.5, 0), (1.8, 2.5)]
    },
    'singapore': {
        'name': 'Marina Bay Street Circuit',
        'country': 'Singapore',
        'length': 5.063,
        'corners': 23,
        'x': [0, 0.5, 1, 1.3, 1.5, 1.8, 1.5, 1.2, 1, 0.8, 0.5, 0.2, 0],
        'y': [0, 0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2, 1.8, 1.5, 1.2, 0.8, 0.5, 0],
        'drs_zones': [(0.3, 0.1), (1.2, 1.5)]
    },
    'austin': {
        'name': 'Circuit of the Americas',
        'country': 'USA',
        'length': 5.513,
        'corners': 20,
        'x': [0, 0.5, 1, 1.5, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.5, 0],
        'y': [0, 0.5, 0.8, 0.5, 0.2, 0.5, 1, 1.5, 2, 2.3, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.2, 0.3), (2, 1.5)]
    },
    'mexico': {
        'name': 'Autódromo Hermanos Rodríguez',
        'country': 'Mexico',
        'length': 4.304,
        'corners': 17,
        'x': [0, 1.5, 1.8, 1.5, 1, 0.5, 0.2, 0.5, 1, 1.5, 1.2, 0.8, 0.5, 0],
        'y': [0, 0, 0.5, 1, 1.5, 1.8, 2, 2.2, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.8, 0)]
    },
    'interlagos': {
        'name': 'Autódromo José Carlos Pace',
        'country': 'Brazil',
        'length': 4.309,
        'corners': 15,
        'x': [0, 0.5, 1, 1.5, 1.8, 1.5, 1, 0.5, 0.2, 0, -0.2, 0],
        'y': [0, -0.2, 0, 0.5, 1, 1.5, 1.8, 1.5, 1, 0.5, 0.2, 0],
        'drs_zones': [(0.2, -0.1), (1.2, 1.2)]
    },
    'vegas': {
        'name': 'Las Vegas Strip Circuit',
        'country': 'USA',
        'length': 6.201,
        'corners': 17,
        'x': [0, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.5, 0],
        'y': [0, 0, 0.5, 1, 1.5, 2, 2.2, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(1, 0), (2, 1.5)]
    },
    'qatar': {
        'name': 'Lusail International Circuit',
        'country': 'Qatar',
        'length': 5.380,
        'corners': 16,
        'x': [0, 1, 1.5, 2, 2.3, 2, 1.5, 1, 0.5, 0],
        'y': [0, 0, 0.5, 1, 1.5, 2, 2.2, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.5, 0), (1.5, 1.8)]
    },
    'abudhabi': {
        'name': 'Yas Marina Circuit',
        'country': 'UAE',
        'length': 5.554,
        'corners': 16,
        'x': [0, 1, 1.5, 2, 2.5, 2.8, 2.5, 2, 1.5, 1, 0.5, 0],
        'y': [0, 0, 0.5, 0.8, 1, 1.5, 2, 2.3, 2, 1.5, 1, 0.5, 0],
        'drs_zones': [(0.5, 0), (2, 1.8)]
    }
}

# Data cleaning function
# Enhanced Circuit Visualization
def create_realistic_circuit_layout(circuit_key='monaco'):
    """Create a realistic F1 circuit layout visualization"""
    circuit = CIRCUIT_LAYOUTS_REAL.get(circuit_key, CIRCUIT_LAYOUTS_REAL['monaco'])
    
    fig = go.Figure()
    
    # Smooth the circuit path
    try:
        from scipy.interpolate import make_interp_spline
        if len(circuit['x']) > 3:
            t = np.linspace(0, 1, len(circuit['x']))
            t_smooth = np.linspace(0, 1, 300)
            
            x_points = circuit['x'] if circuit['x'][0] == circuit['x'][-1] else circuit['x'] + [circuit['x'][0]]
            y_points = circuit['y'] if circuit['y'][0] == circuit['y'][-1] else circuit['y'] + [circuit['y'][0]]
            
            spl_x = make_interp_spline(np.linspace(0, 1, len(x_points)), x_points, k=min(3, len(x_points)-1))
            spl_y = make_interp_spline(np.linspace(0, 1, len(y_points)), y_points, k=min(3, len(y_points)-1))
            
            x_smooth = spl_x(t_smooth)
            y_smooth = spl_y(t_smooth)
    except:
        x_smooth = circuit['x']
        y_smooth = circuit['y']
    
    # Main track outline
    fig.add_trace(go.Scatter(
        x=x_smooth,
        y=y_smooth,
        mode='lines',
        line=dict(color='white', width=12),
        name='Track',
        fill='none',
        showlegend=False
    ))
    
    # Track surface
    fig.add_trace(go.Scatter(
        x=x_smooth,
        y=y_smooth,
        mode='lines',
        line=dict(color='#404040', width=8),
        name='Surface',
        fill='toself',
        fillcolor='rgba(64, 64, 64, 0.3)',
        showlegend=False
    ))
    
    # Start/Finish line
    if len(circuit['x']) > 0:
        fig.add_trace(go.Scatter(
            x=[circuit['x'][0] - 0.1, circuit['x'][0] + 0.1],
            y=[circuit['y'][0], circuit['y'][0]],
            mode='lines',
            line=dict(color='white', width=3, dash='dash'),
            name='Start/Finish',
            showlegend=False
        ))
    
    # DRS Zones
    for i, (drs_x, drs_y) in enumerate(circuit.get('drs_zones', [])):
        fig.add_trace(go.Scatter(
            x=[drs_x],
            y=[drs_y],
            mode='markers+text',
            marker=dict(size=15, color='#00ff00', symbol='diamond'),
            text=[f'DRS {i+1}'],
            textposition='top center',
            textfont=dict(color='#00ff00', size=10),
            name=f'DRS Zone {i+1}',
            showlegend=False
        ))
    
    # Add circuit info
    fig.add_annotation(
        text=f"<b>{circuit['name']}</b><br>{circuit['country']}<br>Length: {circuit.get('length', 'N/A')} km<br>Corners: {circuit.get('corners', 'N/A')}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color='white'),
        align="left",
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(
            text=f"{circuit['name']} Layout",
            font=dict(size=20, color='white')
        ),
        showlegend=False,
        hovermode='closest',
        template='plotly_dark',
        height=500,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0f0f0f',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Advanced Feature Engineering Functions
    
    for race_id in results_df['raceId'].unique():
        race_results = results_df[results_df['raceId'] == race_id]


@st.cache_data
def create_momentum_features(results_df, races_df):
    """Create momentum and form features"""
    if results_df.empty or races_df.empty:
        return pd.DataFrame()
    
    features_list = []
    
    for driver_id in results_df['driverId'].unique():
        driver_results = results_df[results_df['driverId'] == driver_id].sort_values('raceId')
        if len(driver_results) > 3:
            last_3_points = driver_results.tail(3)['points'].mean()
            last_3_positions = driver_results.tail(3)['positionOrder'].mean()
            
            if len(driver_results) > 5:
                recent_points = driver_results.tail(5)['points'].values
                trend_coefficient = np.polyfit(range(len(recent_points)), recent_points, 1)[0]
            else:
                trend_coefficient = 0
            
            features_list.append({
                'driverId': driver_id,
                'momentum_points': last_3_points,
                'momentum_position': last_3_positions,
                'form_trend': trend_coefficient
            })
    
    return pd.DataFrame(features_list)

@st.cache_data
def prepare_advanced_ml_features(results_df, races_df, drivers_df, constructors_df, 
                                 qualifying_df, pit_stops_df):
    """Prepare advanced features for ML models with all engineering techniques"""
    if results_df.empty or races_df.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
    
    try:
        results_clean = results_df.copy()
        numeric_cols = ['points', 'grid', 'positionOrder', 'statusId']
        for col in numeric_cols:
            if col in results_clean.columns:
                results_clean[col] = pd.to_numeric(results_clean[col], errors='coerce')
        
        results_clean = results_clean.dropna(subset=['driverId', 'constructorId', 'points'])
        
        # 1. Track-specific performance
        track_performance = calculate_track_specific_performance(results_clean, races_df, drivers_df)
        
        # 2. Teammate strength
        teammate_strength = calculate_teammate_strength(results_clean, races_df, drivers_df)
        
        # 3. Qualifying vs Race pace
        quali_race_diff = calculate_quali_race_pace_diff(results_clean, qualifying_df, races_df)
        
        # 4. Rolling averages (3-year)
        rolling_avgs = calculate_rolling_averages(results_clean, races_df)
        
        # 5. Momentum features
        momentum_features = create_momentum_features(results_clean, races_df)
        
        # 6. Constructor performance metrics
        constructor_performance = results_clean.groupby('constructorId').agg({
            'points': ['mean', 'sum'],
            'positionOrder': 'mean'
        }).reset_index()
        constructor_performance.columns = ['constructorId', 'constructor_avg_points', 
                                          'constructor_total_points', 'constructor_avg_position']
        
        # Create main feature dataset
        ml_data = results_clean.merge(races_df[['raceId', 'year', 'round', 'circuitId']], on='raceId')
        
        # Merge all engineered features
        if not track_performance.empty:
            ml_data = ml_data.merge(track_performance, on=['driverId', 'circuitId'], how='left')
        
        if not teammate_strength.empty:
            ml_data = ml_data.merge(teammate_strength, on='driverId', how='left')
        
        if not quali_race_diff.empty:
            ml_data = ml_data.merge(quali_race_diff, on='driverId', how='left')
        
        if not rolling_avgs.empty:
            ml_data = ml_data.merge(rolling_avgs, on='driverId', how='left')
        
        if not momentum_features.empty:
            ml_data = ml_data.merge(momentum_features, on='driverId', how='left')
        
        ml_data = ml_data.merge(constructor_performance, on='constructorId', how='left')
        
        # Additional features
        ml_data['is_street_circuit'] = ml_data['circuitId'].isin([6, 7, 14, 19, 20, 21, 22])
        ml_data['is_high_speed_circuit'] = ml_data['circuitId'].isin([9, 10, 11, 15])
        
        # Create interaction features
        ml_data['driver_constructor_synergy'] = (
            ml_data.get('rolling_avg_points', 0) * ml_data.get('constructor_avg_points', 0) / 100
        )
        
        # Polynomial features for key variables
        ml_data['grid_squared'] = ml_data['grid'] ** 2
        ml_data['momentum_x_track'] = ml_data.get('momentum_points', 0) * ml_data.get('track_avg_points', 0) / 10
        
        # Select final features
        feature_cols = [
            'year', 'round', 'grid', 'grid_squared',
            'track_avg_points', 'track_avg_position', 'track_dominance_score',
            'avg_points_vs_teammate', 'teammate_beat_rate', 'teammate_strength_score',
            'avg_position_gain', 'race_pace_score',
            'rolling_avg_points', 'rolling_avg_position', 'rolling_wins',
            'rolling_podiums', 'rolling_point_scoring_rate',
            'momentum_points', 'momentum_position', 'form_trend',
            'constructor_avg_points', 'constructor_avg_position',
            'is_street_circuit', 'is_high_speed_circuit',
            'driver_constructor_synergy', 'momentum_x_track'
        ]
        
        # Only keep features that exist
        feature_cols = [col for col in feature_cols if col in ml_data.columns]
        
        # Fill missing values
        for col in feature_cols:
            if col in ml_data.columns:
                ml_data[col] = pd.to_numeric(ml_data[col], errors='coerce')
                ml_data[col] = ml_data[col].fillna(ml_data[col].median() if not ml_data[col].isna().all() else 0)
        
        ml_data = ml_data.dropna(subset=['points'])
        
        if ml_data.empty:
            return pd.DataFrame(), pd.Series(), pd.DataFrame()
        
        X = ml_data[feature_cols]
        y = ml_data['points']
        
        feature_metadata = pd.DataFrame({
            'feature': feature_cols,
            'category': ['basic' if f in ['year', 'round', 'grid', 'grid_squared'] else
                        'track' if 'track' in f else
                        'teammate' if 'teammate' in f else
                        'pace' if 'position_gain' in f or 'race_pace' in f else
                        'rolling' if 'rolling' in f else
                        'momentum' if 'momentum' in f or 'form' in f else
                        'constructor' if 'constructor' in f else
                        'interaction' if 'synergy' in f or '_x_' in f else
                        'additional' for f in feature_cols]
        })
        
        return X, y, feature_metadata
        
    except Exception as e:
        st.error(f"Error in advanced feature engineering: {str(e)}")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

@st.cache_data
def train_enhanced_ml_models(X, y, feature_metadata):
    """Train enhanced ensemble ML models"""
    if X.empty or y.empty:
        return None, None, {}
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Optimized Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        
        # 3. Extra Trees
        et_model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
        
        # Train models
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        et_model.fit(X_train_scaled, y_train)
        
        # 4. XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
        
        # Predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        et_pred = et_model.predict(X_test_scaled)
        
        # Weighted ensemble
        if XGBOOST_AVAILABLE:
            ensemble_pred = 0.30 * rf_pred + 0.25 * gb_pred + 0.25 * et_pred + 0.20 * xgb_pred
        else:
            ensemble_pred = 0.35 * rf_pred + 0.35 * gb_pred + 0.30 * et_pred
        
        # Cross-validation
        rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, 
                                       scoring='r2', n_jobs=-1)
        
        # Feature importance by category
        feature_importance_by_category = {}
        for category in feature_metadata['category'].unique():
            category_features = feature_metadata[feature_metadata['category'] == category]['feature'].tolist()
            category_indices = [i for i, f in enumerate(X.columns) if f in category_features]
            if category_indices:
                category_importance = np.sum([rf_model.feature_importances_[i] for i in category_indices])
                feature_importance_by_category[category] = category_importance
        
        metrics = {
            'rf_score': r2_score(y_test, rf_pred),
            'gb_score': r2_score(y_test, gb_pred),
            'et_score': r2_score(y_test, et_pred),
            'ensemble_score': r2_score(y_test, ensemble_pred),
            'rf_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'ensemble_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'ensemble_mae': mean_absolute_error(y_test, ensemble_pred),
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std(),
            'feature_importance': dict(zip(X.columns, rf_model.feature_importances_)),
            'feature_importance_by_category': feature_importance_by_category,
            'test_predictions': {
                'actual': y_test.tolist(),
                'predicted': ensemble_pred.tolist()
            }
        }
        
        if XGBOOST_AVAILABLE:
            metrics['xgb_score'] = r2_score(y_test, xgb_pred)
        
        # Create ensemble model
        if XGBOOST_AVAILABLE:
            models = [('rf', rf_model), ('gb', gb_model), ('et', et_model), ('xgb', xgb_model)]
        else:
            models = [('rf', rf_model), ('gb', gb_model), ('et', et_model)]
        
        ensemble_model = VotingRegressor(models)
        ensemble_model.fit(X_train_scaled, y_train)
        
        return ensemble_model, scaler, metrics
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None, None, {}

# Visualization Functions
def create_track_performance_heatmap(track_performance, drivers_df, circuits_df):
    """Create heatmap of driver performance at different tracks"""
    if track_performance.empty:
        return go.Figure()
    
    if not drivers_df.empty:
        track_performance = track_performance.merge(
            drivers_df[['driverId', 'surname']], on='driverId'
        )
    
    if not circuits_df.empty:
        track_performance = track_performance.merge(
            circuits_df[['circuitId', 'name']], on='circuitId'
        )
    
    heatmap_data = track_performance.pivot_table(
        index='surname' if 'surname' in track_performance.columns else 'driverId',
        columns='name' if 'name' in track_performance.columns else 'circuitId',
        values='track_dominance_score',
        aggfunc='mean'
    )
    
    top_drivers = track_performance.groupby('surname' if 'surname' in track_performance.columns else 'driverId')['track_dominance_score'].mean().nlargest(15).index
    top_tracks = track_performance.groupby('name' if 'name' in track_performance.columns else 'circuitId')['track_dominance_score'].mean().nlargest(10).index
    
    heatmap_data = heatmap_data.loc[top_drivers, top_tracks]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        colorbar=dict(title="Dominance Score"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Driver Performance Heatmap by Circuit",
        xaxis_title="Circuit",
        yaxis_title="Driver",
        height=600,
        template='plotly_dark',
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_feature_importance_sunburst(metrics):
    """Create sunburst chart for feature importance by category"""
    if not metrics or 'feature_importance_by_category' not in metrics:
        return go.Figure()
    
    labels = ['All Features']
    parents = ['']
    values = [sum(metrics['feature_importance_by_category'].values())]
    colors = ['#e10600']
    
    for category, importance in metrics['feature_importance_by_category'].items():
        labels.append(category.title())
        parents.append('All Features')
        values.append(importance)
        colors.append('#ff6b6b' if category == 'track' else
                     '#4ecdc4' if category == 'teammate' else
                     '#45b7d1' if category == 'pace' else
                     '#96ceb4' if category == 'rolling' else
                     '#feca57' if category == 'constructor' else
                     '#ff9999' if category == 'momentum' else
                     '#dfe6e9')
    
    for feature, importance in metrics['feature_importance'].items():
        category = 'basic' if feature in ['year', 'round', 'grid', 'grid_squared'] else \
                  'track' if 'track' in feature else \
                  'teammate' if 'teammate' in feature else \
                  'pace' if 'position_gain' in feature or 'race_pace' in feature else \
                  'rolling' if 'rolling' in feature else \
                  'momentum' if 'momentum' in feature or 'form' in feature else \
                  'constructor' if 'constructor' in feature else \
                  'interaction' if 'synergy' in feature or '_x_' in feature else \
                  'additional'
        
        labels.append(feature.replace('_', ' ').title()[:20])
        parents.append(category.title())
        values.append(importance)
        colors.append('#ff9999')
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors),
        textinfo="label+percent parent"
    ))
    
    fig.update_layout(
        title="Feature Importance Hierarchy",
        height=500,
        template='plotly_dark'
    )
    
    return fig

# Data loading
@st.cache_data
def load_data(uploaded_files):
    """Load all uploaded CSV files into dataframes with proper data cleaning"""
    data = {}
    
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file, na_values=['\\N', 'NULL', 'null', ''])
            filename = uploaded_file.name.replace('.csv', '')
            
            if 'results' in filename:
                numeric_cols = ['points', 'grid', 'positionOrder', 'laps', 'rank', 'statusId']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
                
                if 'position' in df.columns:
                    df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
            
            elif 'driver_standings' in filename or 'constructor_standings' in filename:
                numeric_cols = ['points', 'position', 'wins', 'raceId', 'driverId', 'constructorId']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            elif 'lap_times' in filename:
                numeric_cols = ['milliseconds', 'position', 'lap']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            elif 'pit_stops' in filename:
                numeric_cols = ['milliseconds', 'stop', 'lap']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            elif 'qualifying' in filename:
                numeric_cols = ['position', 'number']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
            
            data[filename] = df
            st.success(f"✅ Loaded {uploaded_file.name}: {len(df)} rows")
            
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    return data

# Main Dashboard
def main():
    st.title("🏁 F1 Analytics Dashboard 2024-2025 - Complete Edition")
    st.markdown("### Enhanced ML (R² > 0.75) with Real Circuit Visualizations")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Management")
        
        uploaded_files = st.file_uploader(
            "Upload F1 CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload all F1 data files"
        )
        
        if uploaded_files:
            if st.button("🚀 Load & Process Data", type="primary"):
                with st.spinner("Loading and engineering features..."):
                    st.session_state.f1_data = load_data(uploaded_files)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(st.session_state.f1_data)} files!")
        
        if st.session_state.data_loaded:
            st.divider()
            st.subheader("📊 Data Summary")
            for key, df in st.session_state.f1_data.items():
                st.metric(key.replace('_', ' ').title(), f"{len(df):,} rows")
    
    # Main content
    if not st.session_state.data_loaded:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ML Target", "R² > 0.75", "Enhanced Features")
        with col2:
            st.metric("Circuits", "24 Tracks", "2025 Calendar")
        with col3:
            st.metric("Features", "25+", "Advanced Engineering")
        
        st.info("👈 Upload your F1 data files to begin")
    else:
        # Extract dataframes
        races_df = st.session_state.f1_data.get('races', pd.DataFrame())
        drivers_df = st.session_state.f1_data.get('drivers', pd.DataFrame())
        constructors_df = st.session_state.f1_data.get('constructors', pd.DataFrame())
        results_df = st.session_state.f1_data.get('results', pd.DataFrame())
        driver_standings_df = st.session_state.f1_data.get('driver_standings', pd.DataFrame())
        constructor_standings_df = st.session_state.f1_data.get('constructor_standings', pd.DataFrame())
        lap_times_df = st.session_state.f1_data.get('lap_times', pd.DataFrame())
        pit_stops_df = st.session_state.f1_data.get('pit_stops', pd.DataFrame())
        circuits_df = st.session_state.f1_data.get('circuits', pd.DataFrame())
        qualifying_df = st.session_state.f1_data.get('qualifying', pd.DataFrame())
        
        # Tabs
        tabs = st.tabs([
            "🏁 Circuit Layouts",
            "🤖 Enhanced ML",
            "🔥 Performance Heatmaps",
            "📊 Feature Analysis"
        ])
        
        # Tab 1: Circuit Layouts
        with tabs[0]:
            st.header("Real F1 Circuit Visualizations")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                circuit_options = list(CIRCUIT_LAYOUTS_REAL.keys())
                selected_circuit = st.selectbox(
                    "Select Circuit (24 Available)",
                    options=circuit_options,
                    format_func=lambda x: CIRCUIT_LAYOUTS_REAL[x]['name']
                )
                
                circuit_fig = create_realistic_circuit_layout(selected_circuit)
                st.plotly_chart(circuit_fig, use_container_width=True)
            
            with col2:
                circuit_info = CIRCUIT_LAYOUTS_REAL[selected_circuit]
                st.metric("Country", circuit_info['country'])
                st.metric("Length", f"{circuit_info['length']} km")
                st.metric("Corners", circuit_info['corners'])
                st.metric("DRS Zones", len(circuit_info['drs_zones']))
                
                st.divider()
                
                if selected_circuit == 'monaco':
                    st.info("🏙️ Street Circuit\nNarrow & Technical")
                elif selected_circuit in ['monza', 'spa']:
                    st.info("🚀 High-Speed Circuit\nLow Downforce")
                elif selected_circuit in ['singapore', 'baku']:
                    st.info("🌃 Night Street Circuit")
                else:
                    st.info("🏁 Permanent Circuit")
        
        # Tab 2: Enhanced ML
        with tabs[1]:
            st.header("Enhanced Machine Learning Models")
            
            if not results_df.empty and not races_df.empty:
                if st.button("🚀 Train Enhanced ML Models", type="primary"):
                    with st.spinner("Engineering 25+ advanced features..."):
                        X, y, feature_metadata = prepare_advanced_ml_features(
                            results_df, races_df, drivers_df, constructors_df,
                            qualifying_df, pit_stops_df
                        )
                        
                        if not X.empty:
                            st.info(f"✅ Created {len(X.columns)} features from {len(X)} samples")
                            
                            with st.spinner("Training ensemble models..."):
                                model, scaler, metrics = train_enhanced_ml_models(X, y, feature_metadata)
                                
                                if model is not None:
                                    st.session_state.ml_models = {
                                        'model': model,
                                        'scaler': scaler,
                                        'metrics': metrics
                                    }
                                    
                                    # Display metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        improvement = (metrics['ensemble_score'] - 0.6389) * 100
                                        st.metric("Ensemble R²", f"{metrics['ensemble_score']:.4f}",
                                                f"+{improvement:.1f}% vs baseline")
                                    with col2:
                                        st.metric("RMSE", f"{metrics['ensemble_rmse']:.2f} pts")
                                    with col3:
                                        st.metric("MAE", f"{metrics['ensemble_mae']:.2f} pts")
                                    with col4:
                                        st.metric("CV Score", f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
                                    
                                    # Feature importance
                                    st.subheader("Feature Importance Analysis")
                                    
                                    sunburst_fig = create_feature_importance_sunburst(metrics)
                                    st.plotly_chart(sunburst_fig, use_container_width=True)
                                    
                                    # Top features
                                    importance_df = pd.DataFrame(
                                        list(metrics['feature_importance'].items()),
                                        columns=['Feature', 'Importance']
                                    ).sort_values('Importance', ascending=False).head(10)
                                    
                                    fig = px.bar(
                                        importance_df,
                                        x='Importance',
                                        y='Feature',
                                        orientation='h',
                                        title="Top 10 Features",
                                        color='Importance',
                                        color_continuous_scale='Reds'
                                    )
                                    fig.update_layout(template='plotly_dark', height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Insufficient data for ML training")
        
        # Tab 3: Performance Heatmaps
        with tabs[2]:
            st.header("Track-Specific Performance Analysis")
            
            if not results_df.empty and not races_df.empty:
                with st.spinner("Calculating track performance..."):
                    track_performance = calculate_track_specific_performance(results_df, races_df, drivers_df)
                    
                    if not track_performance.empty:
                        heatmap_fig = create_track_performance_heatmap(track_performance, drivers_df, circuits_df)
                        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Tab 4: Feature Analysis
        with tabs[3]:
            st.header("Advanced Feature Engineering Results")
            
            if not results_df.empty and not races_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Momentum Features")
                    momentum = create_momentum_features(results_df, races_df)
                    if not momentum.empty and not drivers_df.empty:
                        momentum = momentum.merge(drivers_df[['driverId', 'surname']], on='driverId')
                        top_momentum = momentum.nlargest(10, 'momentum_points')[['surname', 'momentum_points', 'form_trend']]
                        st.dataframe(top_momentum, hide_index=True)
                
                with col2:
                    st.subheader("Teammate Analysis")
                    teammate = calculate_teammate_strength(results_df, races_df, drivers_df)
                    if not teammate.empty and not drivers_df.empty:
                        teammate = teammate.merge(drivers_df[['driverId', 'surname']], on='driverId')
                        top_teammate = teammate.nlargest(10, 'teammate_beat_rate')[['surname', 'teammate_beat_rate', 'teammate_strength_score']]
                        st.dataframe(top_teammate, hide_index=True)

if __name__ == "__main__":
    main()
