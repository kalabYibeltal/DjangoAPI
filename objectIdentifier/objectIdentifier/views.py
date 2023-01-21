from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import tensorflow as tf
# import matplotlib as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt 


@api_view(['GET', 'POST'])

def object_identify(request):
    if request.method == 'GET':
        return JsonResponse({"verdict":65}, safe=False)
    if request.method == 'POST':
        return JsonResponse({"verdict":"yes we can"}, safe=False)
    
        image = request.data['image']
        new_model = tf.keras.models.load_model("C:/Users/kalab/OneDrive/Desktop/AI_test/mobilenet_v1_1.0_224.tflite")

    
        img = image.load_img(image, target_size=(200,200,3))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis =0)
        images = np.vstack([x])
        val = new_model.predict(images)
        # if val == 0:
        #     print('50 birr')
        # else:
        #     print('bag')
        print(val)
        return JsonResponse({"verdict":val}, safe=False)