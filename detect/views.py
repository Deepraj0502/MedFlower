import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from django.shortcuts import render
import requests

# Create your views here.
img_url=""

# load model
savedModel = load_model("detect/Model.h5")


def home(request):
    return render(request, "index.html")

def mediuse(request):
    global img_url
    content={}
     # Creating list for mapping
    list_ = ['Daisy', 'Danelion', 'Rose', 'Sunflower', 'Tulip']
    
    test_image = image.load_img("image.jpg", target_size=(224, 224))

    # For show image
    plt.imshow(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Result array
    result = savedModel.predict(test_image)

    # Mapping result array with the main name list
    i = 0
    for i in range(len(result[0])):
        if(result[0][i] == 1):
            print(list_[i])
            content['name'] = list_[i]
            break
    content['image'] = img_url
    return render(request, "mediuse.html",content)

def info(request):
    global img_url
    img_url = request.POST.get("url")
    print(img_url)
    response = requests.get(img_url)
    if response.status_code:
        fp = open('image.jpg', 'wb')
        fp.write(response.content)
        fp.close()
    return render(request,"mediuse.html")

