import streamlit as st
from keras.models import load_model 
import numpy as np 

model = load_model("model.h5")
labels = np.load("labels.npy") 

# Read the content of the CSS file
with open("style.css", "r") as f:
    css_code = f.read()

# Title and input fields
st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

st.title("Welcome to flower prediction app")

a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))

btn = st.button("predict")

if btn:
	pred = model.predict(np.array([a,b,c,d]).reshape(1,-1))
	pred = labels[np.argmax(pred)]
	st.subheader(pred)

	if pred=="Iris-setosa":
		st.image("setosa.jpg")
	elif pred=="Iris-versicolor":
		st.image("versicolor.jpg")
	else:	
		st.image("verginca.jpg")

