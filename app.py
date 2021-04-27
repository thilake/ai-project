import tensorflow as tf
import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.model.load_model('/content/imgclassifier.hdf5')
  return model
  model=load_model()
  st.write('IMAGE CLASSIFICATION')

file = st.file_uploader('Please upload an image',type=["jpg","png"])
import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):

  size = 224,224
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img = np.asarray(image)
  img_reshape = image[np.newaxis,...]
  prediction = model.predict(img.img_reshape)
  return prediction

if file is None:
  st.text = ('Please upload an image file')
else:
  image = Image.open(file)
  st.image = (image, use_column_width=True)
  prediction = import_and_predict(image,model)
  class_name = ['Predator','Alien']
  strings = "This image is most likely to be of:"+class_name[np.argmax(prediction)]
  st.success(string)
 