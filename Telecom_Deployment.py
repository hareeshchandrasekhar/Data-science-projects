# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:19:55 2022

@author: Subash S
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:58:07 2022

@author: Subash S
"""
import numpy as np
import pickle
import streamlit as st
import pandas as pd
#import shap
#import matplotlib.pyplot as plt
import plotly.express as px


st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Customer Churn Prediction App
This app predicts the **Customer Churn Prediction**!
""")
st.write('---')
st.write('**Description of Dataset**')
st.write('**account_length** - Account Length of the Customer')
st.write('**voice_mail_plan** - Voice Mail Plan for the Customer')
st.write('**voice_mail_messages** - Voice Mail Messages for the Customer')
st.write('**day_mins** - Call day Mins Of the Customer')
st.write('**evening_mins** - Call Evening Mins Of the Customer')
st.write('**night_mins** - Call Night Mins Of the Customer')
st.write('**international_mins** - Call NightInternaltion Mins Of the Customer')
st.write('**customer_service_calls** - How Many Times Called to Customer care')
st.write('**international_plan** -  Customer has any International Plan')
st.write('**day_calls** - Day Calls of the Customer')
st.write('**day_charge** - Day Charges of the Customer')
st.write('**evening_calls** - Evening Calls of the Customer')
st.write('**evening_charge** - Evening Charges of the Customer')
st.write('**night_calls** - Night Calls of the Customer')
st.write('**night_charge** - Night Charges of the Customer')
st.write('**international_calls** - International Calls of the Customer')
st.write('**international_charge** - International Charges of the Customer')
st.write('**total_charge** - Total Charges of the Customer')

# Loads the Telecom Dataset
df_telecom = pd.read_csv(r"C:/Users/Subash S/Downloads/Project_1/telecommunications_churn.csv",sep=';')
X =df_telecom.drop('churn',axis=1)
Y = df_telecom[['churn']]

#st.write(df_telecom)

#Visualisation
chart_select = st.sidebar.selectbox(
    label ="Type of chart",
    options=['Scatterplots','Lineplots','Histogram','Boxplot']
)

numeric_columns = list(df_telecom.select_dtypes(['float','int']).columns)

if chart_select == 'Scatterplots':
    st.sidebar.subheader('Scatterplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.scatter(data_frame=df_telecom,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Histogram':
    st.sidebar.subheader('Histogram Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        plot = px.histogram(data_frame=df_telecom,x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Lineplots':
    st.sidebar.subheader('Lineplots Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.line(df_telecom,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Boxplot':
    st.sidebar.subheader('Boxplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.box(df_telecom,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)


# Sidebar
# Header of Specify Input Parameters
#st.sidebar.header('Specify Input Parameters')
st.sidebar.slider('account_length',float(X.account_length.min()), float(X.account_length.max()), float(X.account_length.mean()))
st.sidebar.slider('voice_mail_plan', float(X.voice_mail_plan.min()), float(X.voice_mail_plan.max()), float(X.voice_mail_plan.mean()))
st.sidebar.slider('voice_mail_messages', float(X.voice_mail_messages.min()), float(X.voice_mail_messages.max()), float(X.voice_mail_messages.mean()))
st.sidebar.slider('day_mins', float(X.day_mins.min()), float(X.day_mins.max()), float(X.day_mins.mean()))
st.sidebar.slider('evening_mins', float(X.evening_mins.min()), float(X.evening_mins.max()), float(X.evening_mins.mean()))
st.sidebar.slider('night_mins', float(X.night_mins.min()), float(X.night_mins.max()), float(X.night_mins.mean()))
st.sidebar.slider('international_mins', float(X.international_mins.min()), float(X.international_mins.max()), float(X.international_mins.mean()))
st.sidebar.slider('customer_service_calls', float(X.customer_service_calls.min()), float(X.customer_service_calls.max()), float(X.customer_service_calls.mean()))
st.sidebar.slider('international_plan', float(X.international_plan.min()), float(X.international_plan.max()), float(X.international_plan.mean()))
st.sidebar.slider('day_calls', float(X.day_calls.min()), float(X.day_calls.max()), float(X.day_calls.mean()))
st.sidebar.slider('day_charge', float(X.day_charge.min()), float(X.day_charge.max()), float(X.day_charge.mean()))
st.sidebar.slider('evening_calls', float(X.evening_calls.min()), float(X.evening_calls.max()), float(X.evening_calls.mean()))
st.sidebar.slider('evening_charge', float(X.evening_charge.min()), float(X.evening_charge.max()), float(X.evening_charge.mean()))
st.sidebar.slider('night_calls', float(X.night_calls.min()), float(X.night_calls.max()), float(X.night_calls.mean()))
st.sidebar.slider('night_charge', float(X.night_charge.min()), float(X.night_charge.max()), float(X.night_charge.mean()))
st.sidebar.slider('international_calls', float(X.international_calls.min()), float(X.international_calls.max()), float(X.international_calls.mean()))
st.sidebar.slider('international_charge', float(X.international_charge.min()), float(X.international_charge.max()), float(X.international_charge.mean()))
st.sidebar.slider('total_charge', float(X.total_charge.min()), float(X.total_charge.max()), float(X.total_charge.mean()))

# loading the saved model
loaded_model = pickle.load(open(r'C:\Users\Subash S\Downloads\Project_1\trained_model.sav', 'rb'))


# creating a function for Prediction

def churn_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The Person is not Churn'
    else:
      return 'The Person is Churn'
  
    
  
def main():
    
    
    # giving a title
    st.title('Churn Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Account_lenght = st.text_input('Enter Account Length')
    Voice_mail_plan = st.text_input('Enter Voice Mail Plan')
    Voice_mail_messages = st.text_input('Enter Voioce Mail Messages')
    Day_mins = st.text_input('Enter Day Mins')
    Evening_mins = st.text_input('Enter Evening Mins')
    Night_mins = st.text_input('Enter Night Mins')
    International_mins = st.text_input('Enter International Mins')
    Customer_service_calls = st.text_input('How Many Times Called to Customer care')
    International_plan = st.text_input('Enter Customer has any International Plan')
    Day_calls = st.text_input('Enter Day Calls of the Customer')
    Day_charges = st.text_input('Enter Day Charges of the Customer')
    Evening_calls = st.text_input('Enter Evening Calls of the Customer')
    Evening_charges = st.text_input('Enter Evening Charges of the Customer')
    Night_calls = st.text_input('Enter Night Calls of the Customer')
    Night_charges = st.text_input('Enter Night Charges of the Customer')
    International_calls = st.text_input('Enter International Calls of the Customer')
    International_charges = st.text_input('Enter International Charges of the Customer')
    Total_charges = st.text_input('Enter Total Charges of the Customer')
    
    
    
    # code for Prediction
    churn_predict = ''
    
    # creating a button for Prediction
    
    if st.button('Customer Churn Result'):
        churn_predict = churn_prediction([Account_lenght, Voice_mail_plan, Voice_mail_messages, Day_mins, Evening_mins, Night_mins, International_mins, Customer_service_calls, International_plan, Day_calls, Day_charges, Evening_calls, Evening_charges, Night_calls, Night_charges, International_calls, International_charges, Total_charges])
        
        
    st.success(churn_predict)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
   # Explaining the model's predictions using SHAP values
   # https://github.com/slundberg/shap 
    
#explainer = shap.TreeExplainer(loaded_model)
#shap_values = explainer.shap_values(X)
#if st.button('Show SHAP Graphs'):
#    st.header('Feature Importance')
#   plt.title('Feature importance based on SHAP values')
#    shap.summary_plot(shap_values, X)
#    st.pyplot(bbox_inches='tight')
#    st.write('---')
#   plt.title('Feature importance based on SHAP values')
#    shap.summary_plot(shap_values, X, plot_type="bar")
#    st.pyplot(bbox_inches='tight')