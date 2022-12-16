import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import  pandas_datareader as data
from keras.models import load_model 
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'


st.title('stock trend prediction')

user_input = st.text_input('Enter stock ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#Describing Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())


# Visualizations 

st.subheader('Closing price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('100 and 200 days moving average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig2)


# Splitting the data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)



# loading the model 
model = load_model('keras_model.h5')


# Testing part 
past_100_days = data_training.tail(100)
final_test = past_100_days.append(data_testing,ignore_index=True)
final_test= scaler.transform(final_test)
x_test = []
y_test = []
for i in range(100,final_test.shape[0]):
    x_test.append(final_test[i-100:i])
    y_test.append(final_test[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)


# Making predictions 
y_pred = model.predict(x_test)


# plotting the predictions
scale_factor = 1/ scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor


st.subheader('Predictions vs Original')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original price")
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)













