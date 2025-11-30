import streamlit as st
import pandas as pd
import numpy as np

# Load the trained model
model=joblib.load('crop_yield_model.pkl')
preprocessor=joblib.load('preprocesser.pkl')

st.title('Crop Yield Prediction')
st.write("Predict Yield Using trained machine learning model")

# ---Define input fields---
year = st.number_input('Year', min_value=1900, max_value=2100, value=2020)
area_cultivated = st.number_input('Area Cultivated (in hectares)', min_value=0.0, value=1.0)
production = st.number_input('Production (in kg)', min_value=0.0, value=100.0)

# categorical inputs
region = st.selectbox('Region', ['Tigray', 'Afar', 'Amhara', 'Oromiya', 'Somalia',
       'Benushangul Gumuz', 'S.N.N.P.R', 'Gambela', 'Harari', 'DIRE DAWA'])  # Replace with actual regions
crop_type = st.selectbox('Crop Type', ['Teff', 'Barely', 'Wheat', 'Maize', 'Sorghum', 'Millet', 'Oats'])  # Replace with actual crop types

# ---- Preprocessing ----
# If your model used Label Encoding or OneHot, ensure same transformation here
# For now, assuming label encoding was used:

def encode_region(region):
    mapping = {'Tigray': 0, 'Afar': 1, 'Amhara': 2, 'Oromiya': 3, 'Somalia': 4,
               'Benushangul Gumuz': 5, 'S.N.N.P.R': 6, 'Gambela': 7, 'Harari': 8, 'DIRE DAWA': 9}
    return mapping.get(region, -1)
def encode_crop_type(crop_type):
    mapping = {'Teff': 0, 'Barely': 1, 'Wheat': 2, 'Maize': 3, 'Sorghum': 4, 'Millet': 5, 'Oats': 6}
    return mapping.get(crop_type, -1)

region_encoded = encode_region(region)
crop_type_encoded = encode_crop_type(crop_type)

# combine inputs into a DataFrame using the same column names used during training
input_df = pd.DataFrame({
    'Year': [year],
    'Region': [region_encoded],
    'crop_type': [crop_type_encoded],
    'Area_cultivatedHa': [area_cultivated],
    'Productionkg': [production]
})

# Load preprocessor and (optional) saved encoders so we transform input the same way as training
preprocessor = None
region_encoder = None
crop_encoder = None
try:
    preprocessor = joblib.load('preprocessor.pkl')
except Exception:
    try:
        preprocessor = joblib.load('preprocesser.pkl')
    except Exception:
        preprocessor = None
try:
    region_encoder = joblib.load('region_encoder.pkl')
except Exception:
    region_encoder = None
try:
    crop_encoder = joblib.load('crop_encoder.pkl')
except Exception:
    crop_encoder = None

# ---- Prediction ----
if st.button('Predict Yield'):
    # If we have saved encoders, prefer them to map strings reliably
    if region_encoder is not None and isinstance(region, str):
        try:
            input_df.loc[0, 'Region'] = int(region_encoder.transform([region])[0])
        except Exception:
            pass
    if crop_encoder is not None and isinstance(crop_type, str):
        try:
            input_df.loc[0, 'crop_type'] = int(crop_encoder.transform([crop_type])[0])
        except Exception:
            pass

    # If a preprocessor exists, use it to transform the input to the feature shape expected by the model
    if preprocessor is not None:
        try:
            X_input = preprocessor.transform(input_df)
        except Exception as e:
            st.error(f'Preprocessor transform failed: {e}')
            st.stop()
    else:
        # No preprocessor available: warn and attempt to use raw features (may fail if model expects transformed features)
        st.warning('No preprocessor found — attempting prediction with raw features. If this fails, save and load the preprocessor used during training.')
        X_input = input_df.values

    try:
        prediction = model.predict(X_input)
        st.success(f'Predicted Yield: {prediction[0]:.2f} kg/hectare')
    except Exception as e:
        st.error(f'Model prediction failed: {e}')
