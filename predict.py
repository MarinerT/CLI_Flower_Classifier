#!/usr/bin/python3

import argparse
import json as simplejson
from utils import *
from PIL import Image
import sys

def Main():
    #parse out variables
    parser = argparse.ArgumentParser(description='Image Classifier.')

    #mandatory arguments 
    parser.add_argument('path', help='string; filepath of image')
    parser.add_argument('model', help='.h5 file')
    
    #not mandatory arguments
    parser.add_argument('--top_k', help='integer; the number of top responses', type=int, default=5)
    parser.add_argument('--category_names', help='a json file; map of label to catetgory',action='store_true',default='./label_map.json')

    args = parser.parse_args()
    
    #creating the model
    #loading the MobileNet from TensorFlow Hub
    url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

    feature_extractor = hub.KerasLayer(url, input_shape = (image_size, image_size,3))
    feature_extractor.trainable = False

    # build the model
    model = tf.keras.Sequential([feature_extractor, tf.keras.layers.Dense(102, activation='softmax')])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

    #loading weights
    model.load_weights(args.model)

    #map labels
    with open(args.category_names,'r') as f:
        class_names = simplejson.load(f)
    
    #process image
    im = Image.open(args.path)
    image = np.asarray(im)
    image = process_image(image)
    processed_image = np.expand_dims(image,axis=0)
    
    #make predictions
    predictions = model(processed_image, training=False)
    prob_predictions = predictions[0]
    top_k_probs, top_k_indices = tf.math.top_k(prob_predictions, k=args.top_k)
    probs = top_k_probs.numpy().tolist()
    classes = top_k_indices.numpy().tolist()
    classes = [n+1 for n in classes]
    labels = [class_names[str(n)] for n in classes]
    
    #print outputs
    if args.top_k != 5:
        for _ in range(args.top_k):
            print('\t\u2022' + str(probs[_]) + ':' + str(labels[_]))
            
    else:
        print(labels[np.argmax(probs)],max(probs))

if __name__ == '__main__':
    Main()
