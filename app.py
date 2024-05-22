import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Set the page configuration
st.set_page_config(
    page_title="ERA HDB Resale Price Predictor",
    page_icon="img/era.png",
    layout="wide"
)

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv('datasets/train_edavis.csv') 

df = load_data()

# Filter to only include central region
df = df[df['Region'] == 'Central']

# Rename 'vacancy' to 'vacancy_in_nearest_pri_sch'
df = df.rename(columns={'vacancy': 'vacancy_in_nearest_pri_sch'})

# Define the features to remove
features_to_remove = [
    'central', 'Region', 'mall_within_500m', 'mall_within_1km', 'mall_within_2km', 
    'hawker_within_500m', 'hawker_within_1km', 'hawker_within_2km', 
    '1room_sold', '2room_sold', '3room_sold', '4room_sold', '5room_sold', 
    'exec_sold', 'multigen_sold', 'studio_apartment_sold', 'lower', 'mid', 'upper', 'planning_area'
]

# Remove the specified features from the dataset
df = df.drop(columns=features_to_remove)

# Define the features
categorical_features = ['town', 'flat_type', 'flat_model', 'storey_range', 'mrt_name', 'pri_sch_name', 'sec_sch_name']
numeric_features = ['floor_area_sqft', 'max_floor_lvl', 'total_dwelling_units', 'mall_nearest_distance', 
                    'hawker_nearest_distance', 'hawker_food_stalls', 'hawker_market_stalls', 
                    'mrt_nearest_distance', 'bus_stop_nearest_distance', 'pri_sch_nearest_distance', 
                    'sec_sch_nearest_dist', 'vacancy_in_nearest_pri_sch', 'cutoff_point']

# Creating a new feature
df['age_of_flat'] = df['tranc_year'] - df['hdb_age']
numeric_features.append('age_of_flat')

# Convert the necessary columns to integers
df['hawker_food_stalls'] = df['hawker_food_stalls'].astype(int)
df['hawker_market_stalls'] = df['hawker_market_stalls'].astype(int)
df['vacancy_in_nearest_pri_sch'] = df['vacancy_in_nearest_pri_sch'].astype(int)
df['age_of_flat'] = df['age_of_flat'].astype(int)
df['max_floor_lvl'] = df['max_floor_lvl'].astype(int)
df['total_dwelling_units'] = df['total_dwelling_units'].astype(int)

# Define binary features
binary_features = ['bus_interchange', 'mrt_interchange', 'market_hawker', 'multistorey_carpark']

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Splitting data into features (X) and target (y)
X = df.drop(['resale_price', 'id'], axis=1)
y = df['resale_price']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Fit the model
model_pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(model_pipeline, 'model_pipeline.pkl')

# Load the model
model_pipeline = joblib.load('model_pipeline.pkl')

# Streamlit app
st.title("🏘️ ERA HDB Resale Price Predictor: Trusted by Generations")

# Housing-related features
with st.expander("Housing-related features", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        input_data = {}
        for feature in ['town', 'flat_type', 'flat_model', 'storey_range', 'age_of_flat', 'floor_area_sqft']:
            input_data[feature] = st.selectbox(feature, df[feature].unique()) if feature in categorical_features else st.slider(feature, min_value=int(df[feature].min()), max_value=int(df[feature].max()), value=int(df[feature].mean()))
    with col2:
        for feature in ['max_floor_lvl', 'total_dwelling_units']:
            input_data[feature] = st.slider(feature, min_value=int(df[feature].min()), max_value=int(df[feature].max()), value=int(df[feature].mean()))

# Transport-related features
with st.expander("Transport-related features"):
    col1, col2 = st.columns(2)
    with col1:
        input_data['mrt_name'] = st.selectbox('mrt_name', df['mrt_name'].unique())
    with col2:
        input_data['mrt_nearest_distance'] = st.slider('mrt_nearest_distance', min_value=0, max_value=int(df['mrt_nearest_distance'].max()), value=int(df['mrt_nearest_distance'].mean()))

    col3, col4 = st.columns(2)
    with col3:
        input_data['bus_stop_nearest_distance'] = st.slider('bus_stop_nearest_distance', min_value=0, max_value=int(df['bus_stop_nearest_distance'].max()), value=int(df['bus_stop_nearest_distance'].mean()))
    with col4:
        input_data['bus_interchange'] = st.selectbox('bus_interchange', options=['No', 'Yes'])
        input_data['bus_interchange'] = 1 if input_data['bus_interchange'] == 'Yes' else 0

# Nearby amenities-related features
with st.expander("Nearby amenities-related features"):
    col1, col2 = st.columns(2)
    with col1:
        input_data['market_hawker'] = st.selectbox('market_hawker', options=['No', 'Yes'])
        input_data['market_hawker'] = 1 if input_data['market_hawker'] == 'Yes' else 0
    with col2:
        input_data['multistorey_carpark'] = st.selectbox('multistorey_carpark', options=['No', 'Yes'])
        input_data['multistorey_carpark'] = 1 if input_data['multistorey_carpark'] == 'Yes' else 0

    col3, col4 = st.columns(2)
    with col3:
        input_data['mall_nearest_distance'] = st.slider('mall_nearest_distance', min_value=0, max_value=int(df['mall_nearest_distance'].max()), value=int(df['mall_nearest_distance'].mean()))
    with col4:
        input_data['hawker_nearest_distance'] = st.slider('hawker_nearest_distance', min_value=0, max_value=int(df['hawker_nearest_distance'].max()), value=int(df['hawker_nearest_distance'].mean()))

    col5, col6 = st.columns(2)
    with col5:
        input_data['hawker_food_stalls'] = st.number_input('hawker_food_stalls', min_value=0, value=int(df['hawker_food_stalls'].mean()))
    with col6:
        input_data['hawker_market_stalls'] = st.number_input('hawker_market_stalls', min_value=0, value=int(df['hawker_market_stalls'].mean()))

# School-related features
with st.expander("School-related features"):
    col1, col2 = st.columns(2)
    with col1:
        input_data['pri_sch_name'] = st.selectbox('pri_sch_name', df['pri_sch_name'].unique())
        input_data['sec_sch_name'] = st.selectbox('sec_sch_name', df['sec_sch_name'].unique())
    with col2:
        input_data['pri_sch_nearest_distance'] = st.slider('pri_sch_nearest_distance', min_value=0, max_value=int(df['pri_sch_nearest_distance'].max()), value=int(df['pri_sch_nearest_distance'].mean()))
        input_data['sec_sch_nearest_dist'] = st.slider('sec_sch_nearest_dist', min_value=0, max_value=int(df['sec_sch_nearest_dist'].max()), value=int(df['sec_sch_nearest_dist'].mean()))

    col3, col4 = st.columns(2)
    with col3:
        input_data['cutoff_point'] = st.slider('cutoff_point', min_value=0, max_value=int(df['cutoff_point'].max()), value=int(df['cutoff_point'].mean()))
        input_data['affiliation'] = st.selectbox('affiliation', options=['No', 'Yes'])
        input_data['affiliation'] = 1 if input_data['affiliation'] == 'Yes' else 0
    with col4:
        input_data['vacancy_in_nearest_pri_sch'] = st.number_input('vacancy_in_nearest_pri_sch', min_value=0, value=int(df['vacancy_in_nearest_pri_sch'].mean()))

# Predict button
if st.button('Predict Resale Price'):
    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)
    st.write(f'Predicted Resale Price: ${prediction[0]:,.2f}')

# Disclaimer
st.markdown("<i style='font-size: 12pt;'>Do note that these are close enough predictions, but should not be the exact rule of measure. The information provided is not updated as of May 22, 2024.</i>", unsafe_allow_html=True)
