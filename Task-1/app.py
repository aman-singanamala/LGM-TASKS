import streamlit as st
import numpy as np
import string
import pickle
import sklearn
import joblib
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
st.set_option('deprecation.showfileUploaderEncoding',False) 
model = pickle.load(open('model.pkl','rb'))

model= open("model.pkl", "rb")
svm_clf=joblib.load(model)

def main():
    st.sidebar.header("Iris")
    


    
setosa= Image.open('Images/setosa.png')
versicolor= Image.open('Images/versicolor.png')
virginica = Image.open('Images/virginica.png')

parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.1', '3.5', '1.4', '0.2']

values=[]

#Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
    parameter_input_values.append(values)

input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')

if st.button("Click Here to Classify"):
    prediction = svm_clf.predict(input_variables)
    
    if prediction ==0:
        st.image(setosa)
    elif prediction ==1:
        st.image(versicolor)
    else:
        st.image(virginica)
#if _name_ =='_main_':
 #   main()