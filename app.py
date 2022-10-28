#!/usr/bin/env python
# coding: utf-8

# In[3]:



import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from PIL import Image
import pickle
st.set_page_config(layout="wide")


# In[4]:

# {
#     "primaryColor": "#ffb888"
#     "backgroundColor": "#f3bf9c",
#     "secondaryBackgroundColor": "#484441",
#     "textColor": "#94918f",
#     "font": "#e4dcd7",
# }

header=st.container()
dataset=st.container()
features=st.container()
features1=st.container()
features2=st.container()
features3=st.container()
modelo=st.container()
modelo_neurona=st.container()

with header:
  st.title("Welcome to the predictions lab")
  st.text("In this lab you can costumise your graphs and see the predictions")
  img = Image.open('hospital_pills.jpg')
  st.image(img,width=700, channels='RGB',caption=None)

with dataset:
  st.header("Attendance in emergency at Scotland hospitals dataset")
  st.text("This a case about the attendance in emergency rooms in the hospitals of Scotland")

  emergency_data= pd.read_csv("datos-emergencia-escocia3.csv")
  st.write(emergency_data.head())
    
  # fig = go.Figure(data=go.Table(
  #   header=dict(values=().columns),
  #   fill_color="#FD8E72",
  #   align="center"),
  #   cells=dict(Values, fill_color="#E5ECF6",align="center"))
      



with features:
  st.header("Attendance por año y estaciones")
  st.text("Overview de las distribuciones por año y estaciones")

  sel_col, disp_col = st.columns(2)  

  with sel_col:  
    st.subheader("Attendance en emergencias por año")
    df = pd.read_csv("datos-emergencia-escocia3.csv",parse_dates=[0])
    dfsize2=df.groupby(["year"]).sum()["Sum attendance emergency"].reset_index()
    fig1 = px.bar(dfsize2, y='Sum attendance emergency',color="Sum attendance emergency",x="year",height=400)
    st.plotly_chart(fig1,use_container_width=True)

  with disp_col:
    st.subheader("Attendance en emergencias por estaciones del año")
    dfsize3=df.groupby(["Season2"]).sum()["Sum attendance emergency"].reset_index()
    fig2 = px.pie(dfsize3.reset_index(), values='Sum attendance emergency', names='Season2')
    fig2.update_layout(margin=dict(t=1, b=130, l=0, r=0))
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2)

with features1:
  st.header("Attendance por franjas de tiempo menores")
  st.text("Overview por semanas y meses del año")

  st.markdown("*Esta es una descripción más a detalle que la anterior")

  sel_col, disp_col = st.columns(2) 
  
  with sel_col:
    st.subheader("Attendance por semanas del año")
    dfsize4=df.groupby(["Week number"]).sum()["Sum attendance emergency"].reset_index()
    fig3 = px.bar(dfsize4, y='Sum attendance emergency',color="Sum attendance emergency",x="Week number",height=400)
    st.plotly_chart(fig3)

  with disp_col:
    st.subheader("Attendance por meses del año")
    dfsize5=df.groupby(["month"]).sum()["Sum attendance emergency"].reset_index()
    fig4 = px.bar(dfsize5, y='Sum attendance emergency',color="Sum attendance emergency",x="month",height=400)
    st.plotly_chart(fig4)


with features2:
  st.header("Attendance per location per year before and after COVID")
  st.text("Overview de las attendance antes y luego del COVID")

  st.markdown("*Esta es una descripción para ver como era el flujo de capacidad en emergencias")

  sel_col, disp_col = st.columns(2) 

  with sel_col:
    st.subheader("Attendance per location per year before 2020")
    options = ['Aberdeen Royal Infirmary', 'Forth Valley Royal Hospital',"Glasgow Royal Infirmary","Hairmyres Hospital","Queen Elizabeth University Hospital","Royal Alexandra Hospital","Royal Infirmary Of Edinburgh At Little France","Wishaw General Hospital"]  
    dfafter_2015 = (df['year'] > 2014) & (df['year'] < 2020)
    filtered_df2015=df.loc[dfafter_2015]
    dfsize2015=filtered_df2015.groupby(["year","Location_Name"]).sum()["Sum attendance emergency"].reset_index()
    top_hospitals_2015 = dfsize2015.loc[dfsize2015['Location_Name'].isin(options)]
    fig_top_hospitals_2015 = px.sunburst(top_hospitals_2015, path=['year', 'Location_Name'], values='Sum attendance emergency')
    st.plotly_chart(fig_top_hospitals_2015)


  with disp_col:
    st.subheader("Attendance per location in 2020")
    dfafter_2020 = (df['year'] > 2019) & (df['year'] < 2022)
    filtered_df2020=df.loc[dfafter_2020] 
    dfsize2020=filtered_df2020.groupby(["year","Location_Name"]).sum()["Sum attendance emergency"].reset_index()
    withoutNHS_df = dfsize2020[dfsize2020['Location_Name'] != 'NHSScotland']
    fig_dfsize2020 = px.bar(withoutNHS_df , x='Sum attendance emergency',color="Sum attendance emergency",y="Location_Name",height=400)
    st.plotly_chart(fig_dfsize2020)
    
    
with features3:
  st.header("Animation of attendance per location per year after 2020 without NHSScotland")
  st.text("Overview de las attendance antes del COVID")

  st.markdown("*Esta es una descripción para ver como era el flujo de capacidad en emergencias")
      
  fig2015_animation = px.bar(top_hospitals_2015, x="Location_Name", y="Sum attendance emergency", color="Location_Name",
  animation_frame="year", animation_group="Location_Name", range_y=[1,30000],height=500, width=1000)
  st.plotly_chart(fig2015_animation)

    
    

dfdrop = df.drop(["Week_Ending_Date",'Season2', 'Diferencia atttendance', "Attendance de la semana anterior","Attendance de hace 2 semanas","Attendance de hace 3 semanas","NHS_Board_Code","NHS_Board_Name","Location_Name","Attendance","Number_Over_4_Hours","Percentage_Within_4_Hours","Number_Over_8_Hours","Percentage_Within_8_Hours","Number_Over_12_Hours","Percentage_Within_12_Hours","Location_Code","Data_Source","number location.Location_Name"], axis=1)
train_df = dfdrop.sample(frac=0.8, random_state=9)
test_df = dfdrop.drop(train_df.index)
train_labels = train_df.pop('Sum attendance emergency')
test_labels = test_df.pop('Sum attendance emergency')

model = keras.Sequential([
layers.Dense(64, activation='relu', input_shape=[train_df.shape[1]]),
layers.Dropout(0.3, seed=2),

layers.Dense(64, activation='swish'),
layers.Dense(64, activation='relu'), 
layers.Dropout(0.3, seed=2),
layers.Dense(64, activation='swish'),
layers.Dense(64, activation='relu'),
layers.Dense(64, activation='swish'), 

layers.Dense(1)
])
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001) 

# model.compile(
# loss=tf.keras.losses.MeanSquaredError(), 
# optimizer=optimizer,
# metrics=['mae']
#  )

with modelo:
   st.header("Arquitectura del modelo")

   sel_col1, disp_col1 = st.columns(2)  
   with sel_col1: 
      model.summary(print_fn=st.text)
      st.text("voy a tener 64 conexiones por cada variables de las 12 que hay 832=12x64+64 que son los sesgos, eso en la primera capa")
      st.text("la capa dropout desactivan algunas conexiones aleatoriamente para generalizar mejor, y tenemos 0 es un parametro que no aprende ningún valor")
      st.text("las demas capas son un 64x64+ el sesgo que es 64., que es uno por cada neurona")
      st.text("la última capa es 65 porque e suna sola neurona de 64 variables más sus sesgo, MxN+Sesgo")

   with disp_col1:
      img1 = Image.open('Prediccion_y_labels.jpg')
      st.image(img1,width=800, channels='RGB',caption=None)




new_model = keras.models.load_model('path_to_my_model.h5')
history = pickle.load(open('training_history','rb'))    



with modelo_neurona:
   st.header("Seleccionemos las variables para predecir el attendance:")
   with st.form('Form1'):
         Week_number = st.slider('Week number',1,53)
         month = st.slider('month', 1,12)
         year = st.slider('year',2015,2022)
         Season = st.selectbox('Season',('Winter','Spring',"Summer","Autumn"))
         Attendance_de_la_semana_anterior_en_emergencias = st.slider('Attendance de la semana anterior en emergencias',0,11000)
         Attendance_de_hace_2_semanas_en_emergencias = st.slider('Attendance de hace 2 semanas en emergencias',0,11000)
         
         if Season == 'Winter':
            Season = 0
         if Season == "Spring":
            Season = 1
         if Season == "Summer":
            Season = 2
         if Season == "Autumn":
            Season = 3

         #return [Week_number,month,year,Season,Attendance_de_la_semana_anterior_en_emergencias,Attendance_de_hace_2_semanas_en_emergencias]
   
         Attendance_de_hace_3_semanas_en_emergencias = st.slider('Attendance de hace 3 semanas en emergencias',0,11000)
         Semana_con_fecha_de_fin_de_mes = st.slider("Semana con fecha de fin de mes",0,1)
         Semana_con_holidays = st.slider("Semana con holidays",0,1)
         Año_de_fundación = st.slider("Año de fundación",1729,2017)
         Beds = st.slider("Beds",34,13000)
         Number_location = st.selectbox("Number_location",('Aberdeen Royal Infirmary', 'Balfour Hospital', 'Belford Hospital','Borders General Hospital', 'Caithness General Hospital',"Dr Gray's Hospital", 'Dumfries & Galloway Royal Infirmary','Forth Valley Royal Hospital', 'Galloway Community Hospital','Gilbert Bain Hospital', 'Glasgow Royal Infirmary','Hairmyres Hospital', 'Inverclyde Royal Hospital','Lorn & Islands Hospital', 'Monklands District General Hospital','NHSScotland', 'Ninewells Hospital', 'Perth Royal Infirmary','Queen Elizabeth University Hospital', 'Raigmore Hospital',"Royal Aberdeen Children's Hospital", 'Royal Alexandra Hospital','Royal Hospital For Children','Royal Hospital For Sick Children (Edinburgh)','Royal Infirmary Of Edinburgh At Little France','Southern General Hospital', "St John's Hospital",'University Hospital Ayr', 'University Hospital Crosshouse','Victoria Hospital', 'Victoria Infirmary', 'Western Infirmary','Western Isles Hospital', 'Wishaw General Hospital'))

         if Number_location == 'University Hospital Ayr':
            Number_location = 1
         if Number_location  == "University Hospital Crosshouse":
            Number_location  = 2
         if Number_location  == "Borders General Hospital	":
            Number_location  = 3
         if Number_location  == "Dumfries & Galloway Royal Infirmary":
            Number_location  = 4
         if Number_location  == "Galloway Community Hospital":
            Number_location  = 5
         if Number_location  == "Victoria Hospital":
            Number_location  = 6
         if Number_location  == "Forth Valley Royal Hospital":	
            Number_location  = 7
         if Number_location  == "Aberdeen Royal Infirmary":	
            Number_location  = 8
         if Number_location  == "Dr Gray's Hospital":	
            Number_location  = 9
         if Number_location  == "Royal Aberdeen Children's Hospital":	
            Number_location  = 10
         if Number_location  == "Glasgow Royal Infirmary":	
            Number_location  = 11
         if Number_location  == "Inverclyde Royal Hospital":	
            Number_location  = 12
         if Number_location  == "Royal Alexandra Hospital":	
            Number_location  = 13
         if Number_location  == "Royal Hospital For Children":	
            Number_location  = 14
         if Number_location  == "Southern General Hospital":	
            Number_location  = 15
         if Number_location  == "Victoria Infirmary":	
            Number_location  = 16
         if Number_location  == "Western Infirmary":	
            Number_location  = 17
         if Number_location  == "Belford Hospital":	
            Number_location  = 18
         if Number_location  == "Caithness General Hospital":	
            Number_location  = 19
         if Number_location  == "Lorn & Islands Hospital":	
            Number_location  = 20
         if Number_location  == "Raigmore Hospital":	
            Number_location  = 21
         if Number_location  == "Hairmyres Hospital":	
            Number_location  = 22
         if Number_location  == "Monklands District General Hospital":	
            Number_location  = 23
         if Number_location  == "Wishaw General Hospital":	
            Number_location  = 24
         if Number_location  == "Royal Hospital For Sick Children (Edinburgh)":	
            Number_location  = 25
         if Number_location  == "Royal Infirmary Of Edinburgh At Little France":	
            Number_location  = 26
         if Number_location  == "St John's Hospital":	
            Number_location  = 27
         if Number_location  == "Balfour Hospital":	
            Number_location  = 28
         if Number_location  == "Gilbert Bain Hospital":	
            Number_location  = 29
         if Number_location  == "Ninewells Hospital":	
            Number_location  = 30
         if Number_location  == "Perth Royal Infirmary":	
            Number_location  = 31
         if Number_location  == "Western Isles Hospital":	
            Number_location  = 32
         if Number_location  == "NHSScotland":	
            Number_location  = 33
         if Number_location  == "Queen Elizabeth University Hospital":	
            Number_location  = 34

         
         vector= [Week_number,month,year,Season,Attendance_de_la_semana_anterior_en_emergencias,Attendance_de_hace_2_semanas_en_emergencias, Attendance_de_hace_3_semanas_en_emergencias,Semana_con_fecha_de_fin_de_mes,Semana_con_holidays,Año_de_fundación,Beds,Number_location]
      
         submitted= st.form_submit_button("Predicción")
   #sum_attendance_emergency = tf.constant([variables()+variables1()])
   
   if submitted:
      #si he rellenado el formulario ejecuta lo de lineas abajo
      sum_attendance_emergency = tf.constant([vector])
      prediction = model.predict(sum_attendance_emergency, steps=1)
      pred = [round(x[0]) for x in prediction]



      
      #result= st.button("predict here")
      st.header("Esta es la predicción de el attendance en emergencias de la semana:")
      st.header(pred)







