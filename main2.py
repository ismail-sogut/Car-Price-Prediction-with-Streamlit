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

st.header(" 🏎️🏎️  :red[Car Price Prediction]  🏎️🏎️", divider="blue")

tab_intro, tab_home, tab_model = st.tabs(["intro", "The Data", "Model"])

# TAB INTRO

tab_intro.image("INTRO.jpg")

# TAB HOME


column_dataset, column_graph = tab_home.columns([3,1])

column_dataset.subheader("About the Dataset")
#
# column_graph.markdown("data'da bulunan değişkenler:")
# column_graph.markdown("name")
# column_graph.markdown("selling_price")
# column_graph.markdown("km_driven")
# column_graph.markdown("year")
# column_graph.markdown("fuel")
# column_graph.markdown("seller_type")
# column_graph.markdown("transmission")
# column_graph.markdown("owner")
# column_graph.markdown("fuel_consumption")
# column_graph.markdown("engine")
# column_graph.markdown("max_power")

df = get_data()
column_dataset.dataframe(df, width=1900)

column_dataset.subheader("", divider="red")

column_info = df.shape
column_dataset.markdown(f"Dataframe Size : {column_info}")

column_dataset.subheader("", divider="red")




# TAB MODE

model = get_model()

column_info, column_model = tab_model.columns(2, gap="large")

column_info.subheader("Please Enter the Car Details:")
column_info.subheader("")


user_input_col1, user_input_col2, result_col = tab_model.columns([2, 2, 4])

# ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'fuel_consumption', 'engine', 'max_power', 'NEW_car_age']

km_driven = user_input_col2.number_input("Milage", max_value=999999,  step=1)

fuel1 = user_input_col1.selectbox(label="Fuel Type", options=["Diesel", "Petrol"])
fuel = [0 if fuel1 == "Diesel" else 1]

seller_type1 = user_input_col1.selectbox(label="Seller Type", options=["Dealer", "Individual"])
seller_type = [0 if seller_type1 == "Dealer" else 1]

transmission = user_input_col1.selectbox(label="Transmission", options=["Automatic", "Manuel"])
transmission = [0 if transmission == "Automatic" else 1]

owner1 = user_input_col1.selectbox(label="Owner Type", options=["First Owner","Second Owner or more"])
owner = [0 if owner1 == "First Owner" else 1]

fuel_consumption = user_input_col2.number_input("Fuel Consumption", min_value=0.0, max_value=44.1,  step=0.1)
engine = user_input_col2.number_input("Engine Size", min_value=600, max_value=4444,  step=10)
max_power = user_input_col2.number_input("Max Power - Torq", min_value=32, max_value=400,  step=1)
NEW_car_age = user_input_col2.number_input("Car Age", min_value=0, max_value=44,  step=1)

user_input = pd.DataFrame({"km_driven": km_driven,
                           "fuel": fuel,
                           "seller_type": seller_type,
                           "transmission": transmission,
                           "owner": owner,
                           "fuel_consumption": fuel_consumption,
                           "engine": engine,
                           "max_power": max_power,
                           "NEW_car_age": NEW_car_age}, index=[0])

result_col.markdown("")
result_col.markdown("")
if result_col.button(" **:green[Predict !]**"):
    result = model.predict(user_input)[0]
    result_col.header(f" Your Price: *:orange[{result:.2f}]* *USD*", anchor=False)
    st.snow()