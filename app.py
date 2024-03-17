#!/usr/bin/env python
# coding: utf-8

# In[3]:



import pickle
import streamlit as st
import numpy as np # added
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
from contextlib import redirect_stdout
import io
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
case=st.container()
set=st.container()
separador_index=st.container()
context=st.container()
features=st.container()
features1=st.container()
separador0=st.container()
features2=st.container()
features3=st.container()
separador=st.container()
modelo=st.container()
description=st.container()
separador1=st.container()
modelo_neurona=st.container()

with header:
   st.title("Welcome to the predictions lab")
   st.text("In this lab you can costumise your graphs and see the predictions")
   img = Image.open('hospital_pills.jpg')
   st.image(img,width=700, channels='RGB',caption=None)


with dataset:
   st.header("Attendance in emergency at Scotland hospitals dataset")
   st.text("This a case about the attendance fluctuation in emergency rooms in hospitals of Scotland")

with case:
   st.markdown(f'<h1 style="color:#8CD790;font-size:24px;">{"Part I: The Case"}</h1>', unsafe_allow_html=True)
   st.markdown(f'<h1 style="color:#8CD790;font-size:14px;">{"The case wants to solve the problem of:"}</h1>', unsafe_allow_html=True)
   st.text("- Identify the times of the year when activity increases")
   st.text("- To provide evidence to improve patient care and support Scottish Government policy")

with set:
   emergency_data= pd.read_csv("Hospital_escocia_emergencias_3.csv")
   st.write(emergency_data.head())
      
   # fig = go.Figure(data=go.Table(
   #   header=dict(values=().columns),
   #   fill_color="#FD8E72",
   #   align="center"),
   #   cells=dict(Values, fill_color="#E5ECF6",align="center"))

with separador_index:
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")

with context:
   st.markdown(f'<h1 style="color:#8CD790;font-size:24px;">{"Part II: Stats"}</h1>', unsafe_allow_html=True)
   st.markdown(f'<h1 style="color:#008080;font-size:34px;">{"Some statistical graphs that give us a little context of the assists"}</h1>', unsafe_allow_html=True)

with features:
   st.text("Overview of the distributions by year and seasons")

   sel_col, disp_col = st.columns(2)  

   with sel_col:  
      st.subheader("Attendance in emergencies per year")
      df = pd.read_csv("Hospital_escocia_emergencias_3.csv",parse_dates=[0])
      #numeric_columns = df.select_dtypes(include=np.number).columns.tolist() # added al final
      dfsize2 = df.groupby(["year"], as_index=False)["Sum attendance emergency"].sum() # modified
      # dfsize2=df.groupby(["year"]).sum()["Sum attendance emergency"].reset_index()
      fig1 = px.bar(dfsize2, y='Sum attendance emergency',color="Sum attendance emergency",x="year",height=400)
      st.plotly_chart(fig1,use_container_width=True)

   with disp_col:
      st.subheader("Attendance in emergencies by seasons of the year")
      dfsize3 = df.groupby(["Season2"], as_index=False)["Sum attendance emergency"].sum() # modified
      # dfsize3=df.groupby(["Season2"]).sum()["Sum attendance emergency"].reset_index()
      fig2 = px.pie(dfsize3.reset_index(), values='Sum attendance emergency', names='Season2')
      fig2.update_layout(margin=dict(t=1, b=130, l=0, r=0))
      fig2.update_traces(textposition='inside', textinfo='percent+label')
      st.plotly_chart(fig2)

with features1:
   st.text("Overview by weeks and months of the year")

   sel_col, disp_col = st.columns(2) 
   
   with sel_col:
      st.subheader("Attendance by weeks of the year")
      dfsize4 = df.groupby(["Week number"], as_index=False)["Sum attendance emergency"].sum() # modified
      # dfsize4=df.groupby(["Week number"]).sum()["Sum attendance emergency"].reset_index()
      fig3 = px.bar(dfsize4, y='Sum attendance emergency',color="Sum attendance emergency",x="Week number",height=400)
      st.plotly_chart(fig3)

   with disp_col:
      st.subheader("Attendance by months of the year")
      dfsize5 = df.groupby(["month"], as_index=False)["Sum attendance emergency"].sum() # modified
      # dfsize5=df.groupby(["month"]).sum()["Sum attendance emergency"].reset_index()
      fig4 = px.bar(dfsize5, y='Sum attendance emergency',color="Sum attendance emergency",x="month",height=400)
      st.plotly_chart(fig4)


with separador0:
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")

with features2:
   st.text("Overview of attendance before and after COVID")

   sel_col, disp_col = st.columns(2) 

   with sel_col:
      st.subheader("Attendance per location per year before 2020")
      options = ['Aberdeen Royal Infirmary', 'Forth Valley Royal Hospital',"Glasgow Royal Infirmary","Hairmyres Hospital","Queen Elizabeth University Hospital","Royal Alexandra Hospital","Royal Infirmary Of Edinburgh At Little France","Wishaw General Hospital"]  
      dfafter_2015 = (df['year'] > 2014) & (df['year'] < 2020)
      filtered_df2015=df.loc[dfafter_2015]
      dfsize2015 = filtered_df2015.groupby(["year","Location_Name"], as_index=False)["Sum attendance emergency"].sum() # modified
      # dfsize2015=filtered_df2015.groupby(["year","Location_Name"]).sum()["Sum attendance emergency"].reset_index()
      top_hospitals_2015 = dfsize2015.loc[dfsize2015['Location_Name'].isin(options)]
      fig_top_hospitals_2015 = px.sunburst(top_hospitals_2015, path=['year', 'Location_Name'], values='Sum attendance emergency')
      st.plotly_chart(fig_top_hospitals_2015)


   with disp_col:
      st.subheader("Attendance per location in 2020")
      dfafter_2020 = (df['year'] > 2019) & (df['year'] < 2022)
      filtered_df2020=df.loc[dfafter_2020]
      dfsize2020 = filtered_df2020.groupby(["year","Location_Name"], as_index=False)["Sum attendance emergency"].sum() # modified
      # dfsize2020=filtered_df2020.groupby(["year","Location_Name"]).sum()["Sum attendance emergency"].reset_index()
      withoutNHS_df = dfsize2020[dfsize2020['Location_Name'] != 'NHSScotland']
      fig_dfsize2020 = px.bar(withoutNHS_df , x='Sum attendance emergency',color="Sum attendance emergency",y="Location_Name",height=400)
      st.plotly_chart(fig_dfsize2020)



with features3:
   st.header("Animation of attendance per location per year before 2020 without NHSScotland")
   st.text("Overview of attendance before COVID")

   fig2015_animation = px.bar(top_hospitals_2015, x="Location_Name", y="Sum attendance emergency", color="Location_Name",
   animation_frame="year", animation_group="Location_Name", range_y=[1,30000],height=500, width=1000)
   st.plotly_chart(fig2015_animation)


   #####

dfdrop = df.drop([
   "Week_Ending_Date",'Season2', 'Diferencia atttendance', "Attendance de la semana anterior",
   "Attendance de hace 2 semanas","Attendance de hace 3 semanas","NHS_Board_Code","NHS_Board_Name",
   "Location_Name","Attendance","Number_Over_4_Hours","Percentage_Within_4_Hours",
   "Number_Over_8_Hours","Percentage_Within_8_Hours","Number_Over_12_Hours",
   "Percentage_Within_12_Hours","Location_Code","Data_Source","number location.Location_Name",
   "Location name 2.Fundacion","Semana con fecha de fin de mes","Semana con holidays",
   "Season","month","year","Location name 2.Beds"], axis=1)
train_df = dfdrop.sample(frac=0.8, random_state=9)
test_df = dfdrop.drop(train_df.index)
train_labels = train_df.pop('Sum attendance emergency')
test_labels = test_df.pop('Sum attendance emergency')

model = keras.Sequential([
layers.Dense(80, activation='relu', input_shape=[train_df.shape[1]]),
layers.Dropout(0.3, seed=4),

layers.Dense(64, activation='relu'),
layers.Dense(32, activation='relu'),
layers.Dropout(0.3, seed=2),
layers.Dense(16, activation='relu'),
layers.Dense(8, activation='relu'),
layers.Dense(6, activation='relu'),

layers.Dense(1)
])
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# model.compile(
# loss=tf.keras.losses.MeanSquaredError(),
# optimizer=optimizer,
# metrics=['mae']
#  )

with separador:
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")

with modelo:
   st.markdown(f'<h1 style="color:#8CD790;font-size:24px;">{"Part III: Predictions"}</h1>', unsafe_allow_html=True)
   st.header("Architecture of the neural network model")
   st.markdown(f'<h1 style="color:#008080;font-size:24px;">{"MAE = 39.34"}</h1>', unsafe_allow_html=True)
   st.markdown(f'<h1 style="color:#008080;font-size:18px;">{"Acc=72.49154310960036 Err=27.50845689039964"}</h1>', unsafe_allow_html=True)
   



   sel_col1, disp_col1 = st.columns(2) 
   with sel_col1: 
      with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        summary = buf.getvalue()
        st.write(summary)

   #sel_col1, disp_col1 = st.columns(2)  
   #with sel_col1: 
     #model.summary(print_fn=st.text)

   with disp_col1:
      img1 = Image.open('Prediccion_y_labels.jpg')
      st.image(img1,width=800, channels='RGB',caption=None)


   
with description:
      st.text("I am going to have 80 connections for each variable of the 12 that there are 1040=12x80+80 which are the biases, that in the first layer")
      st.text("the dropout layer deactivates some connections randomly to generalize better, and if we have 0 it is a parameter that does not learn any value")
      st.text("layer two is 64x64, plus the bias is 64, which is one for each neuron")
      st.text("the last layer is 6 because it is a single neuron of 6 variables plus its bias, MxN+Bias")


#new_model = load_model('path_to_my_model_1.h5')#####
#new_model = load_model('path.h5')#####
#new_model = keras.models.load_model('path_to_my_model_1.h5')#####
#new_model = keras.models.load_model('path_to_my_model_fitting.keras')#####
new_model = load_model('path_to_my_model_fitting4.keras')#####
history = pickle.load(open('training_history','rb'))####    

with separador1:
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")
   st.text("                                                                            ")
######
with modelo_neurona:
   st.markdown(f'<h1 style="color:#8CD790;font-size:24px;">{"Part IV: The lab"}</h1>', unsafe_allow_html=True)
   #st.header("Seleccionemos las variables para predecir el attendance:")
   st.markdown(f'<h1 style="color:#008080;font-size:34px;">{"Lets select the variables to predict attendance by hospital"}</h1>', unsafe_allow_html=True)
   sel_col2,cent_col2,disp_col2 = st.columns(3)  

   with sel_col2: 
      st.text("Aberdeen Royal Infirmary: 32-283")
      st.text("Balfour Hospital: 0-17")
      st.text("Belford Hospital: 0-39")
      st.text("Borders General Hospital: 5-177")
      st.text("Caithness General Hospital: 0-29")
      st.text("Dr Gray's Hospital: 2-80")
      st.text("Dumfries & Galloway Royal Infirmary: 5-169")
      st.text("Forth Valley Royal Hospital: 15-716")
      st.text("Galloway Community Hospital: 1-55")
      st.text("Gilbert Bain Hospital: 0-27")
      st.text("Glasgow Royal Infirmary	: 60-751")
      st.text("Hairmyres Hospital: 22-640")

   with cent_col2: 
      st.text("Inverclyde Royal Hospital: 8-260")
      st.text("Lorn & Islands Hospital: 0-17")
      st.text("Monklands District General Hospital: 9-383")
      st.text("NHSScotland: 631-10264")
      st.text("Ninewells Hospital: 1-140")
      st.text("Perth Royal Infirmary: 0-100")
      st.text("Queen Elizabeth University Hospital: 24-1289")
      st.text("Raigmore Hospital: 10-152")
      st.text("Royal Aberdeen Children's Hospital: 0-17")
      st.text("Royal Alexandra Hospital: 62-666")
      st.text("Royal Hospital For Children: 0-393")
      st.text("Royal Hospital For Sick Children (Edinburgh): 2-146")

   with disp_col2: 
      st.text("Royal Infirmary Of Edinburgh At Little France: 47-1504")
      st.text("Southern General Hospital: 66-313")
      st.text("St John's Hospital: 13-386")
      st.text("University Hospital Ayr: 7-411")
      st.text("University Hospital Crosshouse: 11-407")
      st.text("Victoria Hospital: 16-310")
      st.text("Victoria Infirmary: 91-434")
      st.text("Western Infirmary: 95-669")
      st.text("Western Isles Hospital: 0-9")
      st.text("Wishaw General Hospital: 29-540")

   #with location:
      #if Number_location = st.selectbox ("Number_location",('Aberdeen Royal Infirmary', 'Balfour Hospital', 'Belford Hospital','Borders General Hospital', 'Caithness General Hospital',"Dr Gray's Hospital", 'Dumfries & Galloway Royal Infirmary','Forth Valley Royal Hospital', 'Galloway Community Hospital','Gilbert Bain Hospital', 'Glasgow Royal Infirmary','Hairmyres Hospital', 'Inverclyde Royal Hospital','Lorn & Islands Hospital', 'Monklands District General Hospital','NHSScotland', 'Ninewells Hospital', 'Perth Royal Infirmary','Queen Elizabeth University Hospital', 'Raigmore Hospital',"Royal Aberdeen Children's Hospital", 'Royal Alexandra Hospital','Royal Hospital For Children','Royal Hospital For Sick Children (Edinburgh)','Royal Infirmary Of Edinburgh At Little France','Southern General Hospital', "St John's Hospital",'University Hospital Ayr', 'University Hospital Crosshouse','Victoria Hospital', 'Victoria Infirmary', 'Western Infirmary','Western Isles Hospital', 'Wishaw General Hospital')):
               #if Number_location == 'University Hospital Ayr':
                  #Attendance_de_la_semana_anterior_en_emergencias = st.slider('Attendance de la semana anterior en emergencias',32,283)

   with st.form('Form1'):
         Number_location = st.selectbox("Number_location",('Aberdeen Royal Infirmary', 'Balfour Hospital', 'Belford Hospital','Borders General Hospital', 'Caithness General Hospital',"Dr Gray's Hospital", 'Dumfries & Galloway Royal Infirmary','Forth Valley Royal Hospital', 'Galloway Community Hospital','Gilbert Bain Hospital', 'Glasgow Royal Infirmary','Hairmyres Hospital', 'Inverclyde Royal Hospital','Lorn & Islands Hospital', 'Monklands District General Hospital','NHSScotland', 'Ninewells Hospital', 'Perth Royal Infirmary','Queen Elizabeth University Hospital', 'Raigmore Hospital',"Royal Aberdeen Children's Hospital", 'Royal Alexandra Hospital','Royal Hospital For Children','Royal Hospital For Sick Children (Edinburgh)','Royal Infirmary Of Edinburgh At Little France','Southern General Hospital', "St John's Hospital",'University Hospital Ayr', 'University Hospital Crosshouse','Victoria Hospital', 'Victoria Infirmary', 'Western Infirmary','Western Isles Hospital', 'Wishaw General Hospital'))

         if Number_location == 'University Hospital Ayr':
            Number_location = 1
         if Number_location  == "University Hospital Crosshouse":
            Number_location  = 2
         if Number_location  == "Borders General Hospital":
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
         
         
         Week_number = st.slider('Week number',1,53)
         Attendance_de_la_semana_anterior_en_emergencias = st.slider('Attendance de la semana anterior en emergencias',0,11000)
         Attendance_de_hace_2_semanas_en_emergencias = st.slider('Attendance de hace 2 semanas en emergencias',0,11000)

         #return [Week_number,month,year,Season,Attendance_de_la_semana_anterior_en_emergencias,Attendance_de_hace_2_semanas_en_emergencias]
   
         Attendance_de_hace_3_semanas_en_emergencias = st.slider('Attendance de hace 3 semanas en emergencias',0,11000)
         Pandemic = st.slider("Pandemic",0,1)

         #Tengo que standarizar este vector
         vector= [Week_number,Attendance_de_la_semana_anterior_en_emergencias,Attendance_de_hace_2_semanas_en_emergencias, Attendance_de_hace_3_semanas_en_emergencias,Number_location,Pandemic]

         
         

         submitted= st.form_submit_button("Predicción")
   #sum_attendance_emergency = tf.constant([variables()+variables1()])
   
   if submitted:
      #si he rellenado el formulario ejecuta lo de lineas abajo
      vector
      vector_mean=[27.860594,141.950019,141.288963,140.738147,17.318515,0.017217]
      vector_std=[14.857525,574.830164,569.924359,569.448320,10.095289,0.130088]
      product_mean = list(map(lambda x,y: x-y ,vector,vector_mean))
      product_mean 
      product_std = list(map(lambda x,y: x/y ,product_mean,vector_std))
      product_std
      sum_attendance_emergency = tf.constant([product_std])
      prediction = model.predict(sum_attendance_emergency, steps=1)
      prediction_final= (prediction*124.734983)+87.417309
      pred = [round(x[0]) for x in prediction_final]



      
      #result= st.button("predict here")
      st.header("Esta es la predicción de el attendance en emergencias de la semana:")
      st.header(pred)







