import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('Car Price Prediction')
df=pd.read_csv('cleaned_car_data.csv')
df

# ---------------------------------------------------------------------------------------------
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 1)


X_train = df_train[['wheelbase', 'carlength', 'carwidth', 'carheight', 'enginesize',
       'boreratio', 'stroke', 'compressionratio', 'horsepower', 
       'diesel', 'rear', 'turbo']]
y_train = df_train['price']
# df_test_predict = X_train.iloc[0:1]
lm = LinearRegression()
model = lm.fit(X_train, y_train)

# -----------------------------------------------------------------------------------------------

with st.form(key='my_form'):
    wheelbase = st.number_input('Wheel base')
    carlength = st.number_input('Car length')
    carwidth = st.number_input('Car width')
    carheight = st.number_input('Car height')
    enginesize = st.number_input('Engine size')
    boreratio = st.number_input('Bore ratio')
    stroke = st.number_input('Stroke')
    compressionratio = st.number_input('Compression ratio')
    horsepower = st.number_input('Horse power')
    diesel = st.number_input("Diesel (0 or 1)", 0, 1)
    rear = st.number_input("Rear (0 or 1)", 0, 1)
    turbo = st.number_input("Turbo (0 or 1)", 0, 1)
    
    # Create DataFrame for user input
    user_input = pd.DataFrame({
        'wheelbase': [wheelbase],
        'carlength': [carlength],
        'carwidth': [carwidth],
        'carheight': [carheight],
        'enginesize': [enginesize],
        'boreratio': [boreratio],
        'stroke': [stroke],
        'compressionratio': [compressionratio],
        'horsepower': [horsepower], 
        'diesel': [diesel],
        'rear': [rear],
        'turbo': [turbo]
    })
    
    # On form submission, predict the price
    if st.form_submit_button("Predict"):
        predicted_price = model.predict(user_input)
        st.write(f"Predicted price: {predicted_price[0]:.2f}")
    else:
        st.write("Provide the required data and press the button to predict the price.")

