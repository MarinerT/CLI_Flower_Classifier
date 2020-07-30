#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

#variables for another time

image_size = 224

#formatting the image for processing (normalize pixels and changing shape to (224,224)
def process_image(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image,(image_size,image_size)) 
  image /= 255
  return image.numpy()


def predict(image_path, model, top_k):
  
  im = Image.open(image_path)
  image = np.asarray(im)
  image = process_image(image)
  processed_image = np.expand_dims(image,axis=0)
  
  '''  
  # loading the MobileNet from TensorFlow Hub
  url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

  feature_extractor = hub.KerasLayer(url, input_shape = (image_size, image_size,3))
  feature_extractor.trainable = False

  # build the model
  model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(102, activation='softmax')])

  model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])

  #loading weights
  model.load_weights(model_given)
  ''' 
  
  #make predictions
  predictions = model(processed_image, training=False)
  prob_predictions = predictions[0]

  #retur predictions & probabilities
  top_k_probs, top_k_indices = tf.math.top_k(prob_predictions, k=top_k)
  probs = top_k_probs.numpy().tolist()
  classes = top_k_indices.numpy().tolist()
  classes = [n+1 for n in classes]
  return probs, classes
