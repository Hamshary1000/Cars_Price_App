import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Cars Prices Prediction App", layout="wide", page_icon="🚗")


#loading the encoding files
typeColumn_encoding=joblib.load('type_mapping.pkl')
modelColumn_encoding=joblib.load('model_mapping.pkl')
data=pd.read_csv("car_prices-cleaned.csv")
df=data.copy()


with st.sidebar:

       st.sidebar.image('car logo.png')
       st.sidebar.subheader("Cars Prices Analysis and prediction app ")
       st.sidebar.write("")
       selected_fuel = st.sidebar.selectbox("Select Fuel Type:", options=["All"] + list(df['fuel'].unique()), index=0, key='sidebar_fuel_select')
       selected_seller_type = st.sidebar.selectbox("Select Seller Type:", options=["All"] + list(df['seller_type'].unique()), index=0, key='sidebar_seller_type')
       selected_transmission = st.sidebar.selectbox("Select transmission Type:", options=["All"] + list(df['transmission'].unique()), index=0, key='sidebar_transmission')
       selected_owner = st.sidebar.selectbox("Select owner Type:", options=["All"] + list(df['owner'].unique()), index=0, key='sidebar_owner')
       st.write("")
       st.sidebar.markdown("Made by [Mohamed Elhamshary 🐱‍👤](https://github.com/Hamshary1000)")












# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📈 Analysis", "🤖 Prediction"])



# Home
with tab1:

       #setting up the context 
       st.title("Cars Prices Analysis & Prediction App")
       st.write('Are you curious about the potential market price of a car? This app allows you to predict the resale price of vehicles using machine learning!Simply input the cars details like its model, year, engine size, and more.  Our algorihm will provide an accurate price estimate based on historical data Whether you are buying or selling, this tool can help you make informed decisions')
       st.image('car logo.png')



# Analysis
# Filter the DataFrame based on selected filters
df_filtered = df.copy()

if selected_fuel != "All":
    df_filtered = df_filtered[df_filtered['fuel'] == selected_fuel]

if selected_seller_type != "All":
    df_filtered = df_filtered[df_filtered['seller_type'] == selected_seller_type]

if selected_transmission != "All":
    df_filtered = df_filtered[df_filtered['transmission'] == selected_transmission]

if selected_owner != "All":
    df_filtered = df_filtered[df_filtered['owner'] == selected_owner]

# Tab 2 visuals
with tab2:

       # Information Cards
       card1, card2, card3, card4 = st.columns((2,2,2,4))
       avg_sellling=int(df['selling_price'].mean())
       avg_kmDriven=int(df['km_driven'].mean())
       count_types=df['type'].nunique()
       Highest_selling_model=df['model'].value_counts().sort_values(ascending=False).index[0]

       # Show The Cards
       card1.metric("Average Price", f"{avg_sellling}")
       card2.metric("Average KM Driven", f"{avg_kmDriven}")
       card3.metric("Number of types", f"{count_types}")
       card4.metric("Highest Selling Model", f"{Highest_selling_model}")

       st.write('')
       st.write('')

              


       visual1, visual2 = st.columns((5, 5))
       with visual1:
              # Visual 1: Average Selling Price by Fuel
              df_fuel = df_filtered.groupby('fuel')['selling_price'].mean()
              fig1 = px.bar(df_fuel, x=df_fuel.index, y=df_fuel.values, title="Average Selling Price by Fuel")
              fig1.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
              st.plotly_chart(fig1)

              # Visual 2: Average Selling Price by Owner
              df_owner = df_filtered.groupby('owner')['selling_price'].mean()
              fig2 = px.bar(df_owner, x=df_owner.values, y=df_owner.index, title="Average Selling Price by Owner",orientation='h')
              fig2.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis=dict(tickformat=""))
              st.plotly_chart(fig2)
       
       with visual2:
              # Visual 3: Average Selling Price by Seller Type
              df_seller_type = df_filtered['seller_type'].value_counts()
              fig3 = px.pie(df_seller_type, names=df_seller_type.index,values=df_seller_type.values,hole=0.4, title="Seller Type with the highest sales")
              fig3.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
              fig3.update_layout(
              legend=dict(
                     orientation="h",  # Horizontal legend
                     yanchor="top",  # Position the legend at the bottom
                     y=0,  # Shift the legend above the chart
                     xanchor="center",  # Align the legend in the center
                     x=0.5  # Center the legend horizontally
              )
              )
              st.plotly_chart(fig3)

              # Visual 4: Average Selling Price by Transmission
              df_transmission = df_filtered.groupby('transmission')['selling_price'].mean()
              fig4 = px.bar(df_transmission, x=df_transmission.index, y=df_transmission.values, title="Average Selling Price by Transmission")
              fig4.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
              st.plotly_chart(fig4)  

              
       st.write('')
       st.header('Average Price over years')
       ans3=df_filtered.groupby('year')['selling_price'].mean()
       fig = px.line(ans3, 
                     x=ans3.index, 
                     y=ans3.values, 
                     labels={'value': 'Avg Selling price', 'year': 'Year'}
                     )
       fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
       st.plotly_chart(fig, use_container_width=True)


       st.write('')
       st.header('Top 10 selling models')
       top_models = df['model'].value_counts().head(10).index
       top_models_df = df[df['model'].isin(top_models)]
       top_models_df = top_models_df.groupby('model').head(1)  
       st.dataframe(top_models_df, use_container_width=True)





       




              







# Prediction
with tab3:
       

       st.write('')
       st.write("Fill-in the following values to predict the Car Price")
       st.write('')

       Year=st.slider('year', 1991, 2020)

       col1, col2 = st.columns((5,5))       
       with col1:    
              #setting up the features buttons
              
              km_driven=st.number_input('Kilo Meters Driven')
              Fuel = st.selectbox('Fuel', ['Diesel', 'Petrol','CNG','LPG'])
              Seller_Type = st.selectbox('Seller Type', ['Individual', 'Dealer','Trustmark Dealer'])
              Transmission=st.selectbox('Transmission', ['Manual', 'Automatic'])
              Owner=st.selectbox('Owner', ['First Owner', 'Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'])
              Seats=st.selectbox('Number of Seats',[4,5,6,7,8])
       with col2:

              Mileage_value_kmpl=st.number_input('Mileage value kmpl')	
              Engine_cc=st.number_input('Engine (cc)')	
              Max_power_bhp=st.number_input('Max power (bhp)')	
              Torque_value=st.number_input('Torque value')

              Type=st.selectbox('Type',list(typeColumn_encoding.keys()))
              Model=st.selectbox("Select a Car Model", list(modelColumn_encoding.keys()))

       #the prediction button
       btn = st.button("Submit 👌")

       if btn == True :

              # loading model and scalers
              ml_model =joblib.load('kn_model.pkl')
              scaler =joblib.load('x_scaled.pkl')
              target_scaler =joblib.load('y_scaled.pkl')
              modelColumn_encoding=joblib.load('model_mapping.pkl')
              typeColumn_encoding=joblib.load('type_mapping.pkl')


              # encoding and scaling input
              Transmission_mapping = {'Automatic': 0, 'Manual': 1}
              Fuel_mapping = {'Diesel': 1, 'Petrol': 3,'CNG':0,'LPG':2}
              Seller_Type_mapping = {'Individual': 1,'Dealer':0,'Trustmark Dealer':2}
              Owner_mapping ={'First Owner':1, 'Second Owner':2,'Third Owner':3,'Fourth & Above Owner':4,'Test Drive Car':5}

              # Mapping the features
              Fuel_encoded = Fuel_mapping [Fuel]
              Seller_Type_encoded = Seller_Type_mapping [Seller_Type]
              Transmission_encoded = Transmission_mapping [Transmission]
              Owner_encoded = Owner_mapping [Owner]
              Type_encoded=typeColumn_encoding[Type]
              Model_encoded=modelColumn_encoding[Model]


              input_data=np.array([[Year,km_driven,Fuel_encoded,Seller_Type_encoded,Transmission_encoded,
                                   Owner_encoded,Seats,Mileage_value_kmpl,
                                   Engine_cc,Max_power_bhp,Torque_value,Type_encoded,Model_encoded]])

              input_data_scaled = scaler.transform(input_data)
              prediction_scaled = ml_model.predict(input_data_scaled)
              #pred_org=target_scaler.inverse_transform(prediction_scaled.reshape(-1,1))
              st.success(prediction_scaled)

 




   
        
