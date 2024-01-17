import streamlit as st
import joblib
import pandas as pd
st.set_page_config(layout="wide")

@st.cache_data

def get_data():
    df = pd.read_excel("new_car_file3.xlsx")
    return df

def get_model():
    model = joblib.load("price_pred.joblib")
    return model

st.header(" 🟡 :red[Car Price Prediction]")

model = get_model()

column_info, column_model = st.columns(2, gap="large")

column_info.subheader("Please enter the car detailes below:")


user_input_col1, user_input_col2, result_col = st.columns([2, 2, 4])

# ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'fuel_consumption', 'engine', 'max_power', 'NEW_car_age']

km_driven = user_input_col2.number_input("km giriniz", max_value=999999,  step=1)
fuel = user_input_col1.selectbox(label="yakıt türü seçiniz (0=Diesel, 1=Petrol)", options=["0", "1"])
seller_type = user_input_col1.selectbox(label="satıcı türü seçiniz (0=Dealer, 1=Individual)", options=["0", "1"])
transmission = user_input_col1.selectbox(label="vites türü seçiniz (0=Automatic, 1=Manuel)", options=["0", "1"])
owner = user_input_col1.selectbox(label="kaçıncı sahibi seçiniz (0=First Owner, 1=Second Owner or more)", options=["0","1"])
fuel_consumption = user_input_col2.number_input("yakıt tüketimi giriniz", min_value=0.0, max_value=44.1,  step=0.1)
engine = user_input_col2.number_input("motor hacmi giriniz", min_value=600, max_value=4444,  step=10)
max_power = user_input_col2.number_input("motor beygir gücü giriniz", min_value=32, max_value=400,  step=1)
NEW_car_age = user_input_col2.number_input("araç yaşı giriniz", min_value=0, max_value=44,  step=1)

user_input = pd.DataFrame({"km_driven": km_driven,
                           "fuel": fuel,
                           "seller_type": seller_type,
                           "transmission": transmission,
                           "owner": owner,
                           "fuel_consumption": fuel_consumption,
                           "engine": engine,
                           "max_power": max_power,
                           "NEW_car_age": NEW_car_age}, index=[0])

if user_input_col2.button("Predict!"):
    result = model.predict(user_input)[0]
    result_col.header(f"Your predicted Price is: :orange[{result:.2f}]!", anchor=False)
    st.snow()