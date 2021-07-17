import streamlit as st 
from PIL import Image
import pickle
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)

html_temp = """
   <div class="" style="background-color:gray;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Addition and Substraction on Image
         """
         )
file = st.file_uploader("Please upload image", type=("jpg", "png"))

import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data):
  #img = image.load_img(image_data, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)

   img1=cv2.imread(file,1)
   img2=np.ones(img1.shape, dtype="uint8")*100
   #cv2_imshow(img1)
   #cv2_imshow(img2)
   #@title Mathematical Operations on Images {run:"auto"} 
   Operation = '-' #@param ["+", "-"] {allow-input: true}
   if Operation=='+':
     img=img1+img2
   if Operation=='-':
     img=img1-img2

   print('Orignal Image:')
   cv2_imshow(img1)
   print('Operated Image:')
   cv2_imshow(img)
   st.image(image_data, use_column_width=True)
   return 0

if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Apply Addition on image"):
  result=import_and_predict(img)

if st.button("Apply Substraction on image"):
  result=import_and_predict(image)
  
if st.button("About"):
  st.header(" Aachal Kala")
  st.subheader("Student, Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
