import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

st.title("CropSeer")
st.sidebar.title("CropSeer: Foresee the Harvest")

st.sidebar.markdown(" An innovative solution that will help local governments and farmers make informed decisions by using data to predict the season's harvest.")

DATA_URL1 = ("preprocessed_data_rabi.csv")
DATA_URL2 = ("transformed_data_kharif.csv")

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL1)
    return data

data = load_data()
if st.sidebar.checkbox("Show Rabi data", False):
    st.write(data)

@st.cache_data(persist=True)
def load_data():
    data1= pd.read_csv(DATA_URL2)
    return data1

data1 = load_data()
if st.sidebar.checkbox("Show Kharif data", False):
    st.write(data1)


selected_dataset = st.sidebar.radio("Select Dataset", ("Rabi", "Kharif"))


if selected_dataset == "Rabi":
    dist_names = data["Dist Name"].unique()
    crop_names = data["Crops"].unique()

    
    selected_dist_name = st.sidebar.selectbox("Select District", dist_names)
    selected_crop_name = st.sidebar.selectbox("Select Crop", crop_names)

    
    filtered_data = data[
        (data["Dist Name"] == selected_dist_name) & (data["Crops"] == selected_crop_name)
    ]

    if not filtered_data.empty:
       
        fig = px.line(filtered_data, x="Year", y=["Area", "Irrigation"], title="Irrigation & Area Trend")
        st.plotly_chart(fig)
    else:
        st.write("No data found for the given Crop and District.")

else:
    dist_names = data1["Dist Name"].unique()
    crop_names = data1["Crops"].unique()

    
    selected_dist_name = st.sidebar.selectbox("Select District", dist_names)
    selected_crop_name = st.sidebar.selectbox("Select Crop", crop_names)

    
    filtered_data = data1[
        (data1["Dist Name"] == selected_dist_name) & (data1["Crops"] == selected_crop_name)
    ]

    if not filtered_data.empty:
        
        fig = px.line(filtered_data, x="Year", y=["Area", "Irrigation"], title="Irrigation & Area Trend")
        st.plotly_chart(fig)
    else:
        st.write("No data found for the given Crop and District.")


def transform_data(data):
    transformer = ColumnTransformer(transformers=[('tnf', OneHotEncoder(sparse_output=False, drop='first'), ['Dist Name', 'Crops'])], remainder='passthrough')
    return transformer, transformer.fit_transform(data.drop(columns=["Production"]))

transformer_rabi, tdata_rabi = transform_data(data)
transformer_kharif, tdata_kharif = transform_data(data1)

@st.cache_data(persist=True)
def load_tdata(tdata):
    return np.array(tdata)

t_data_rabi = load_tdata(tdata_rabi)
t_data_kharif = load_tdata(tdata_kharif)

def train_model(t_data, target):
    x_train, x_test, y_train, y_test = train_test_split(t_data, target, test_size=0.2, random_state=6)
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(x_train, y_train)
    return rf_regressor

rf_regressor_rabi = train_model(t_data_rabi, data["Production"])
rf_regressor_kharif = train_model(t_data_kharif, data1["Production"])

st.subheader("Enter Values for Prediction")


# Individual feature inputs
year_input = st.text_input("Year:")
if selected_dataset == "Rabi":
    dist_name_input = st.selectbox("District:", data['Dist Name'].unique())
    crop_name_input = st.selectbox("Crop:", data["Crops"].unique())
else:
    dist_name_input = st.selectbox("District:", data1["Dist Name"].unique())
    crop_name_input = st.selectbox("Crop:", data1['Crops'].unique())

area_input = st.number_input("Area (1000ha):")
irrigation_input = st.number_input("Irrigation Area (1000ha):")
precipitation_input = st.number_input("Precipitation (mm):")
temperature_max_input = st.number_input("Temperature(C) (MAX):")
temperature_min_input = st.number_input("Temperature(C) (MIN):")

# Prediction button
predict_button = st.button("Forecast Production")

# Prediction logic (assuming a trained model exists)
if predict_button:
    new_data_pred = pd.DataFrame({
        'Year': [year_input],
        'Dist Name': [dist_name_input],
        'Crops': [crop_name_input],
        'Area': [area_input],
        'Irrigation': [irrigation_input],
        'Precipitation': [precipitation_input],
        'Temperature (MAX)': [temperature_max_input],
        'Temperature (MIN)': [temperature_min_input]
    })

    if selected_dataset == "Rabi":
        new_data_transformed = transformer_rabi.transform(new_data_pred)
        prediction = rf_regressor_rabi.predict(new_data_transformed)[0]
    else:
        new_data_transformed = transformer_kharif.transform(new_data_pred)
        prediction = rf_regressor_kharif.predict(new_data_transformed)[0]

    st.write(f"Forecasted Production (1000 tons): {prediction}")