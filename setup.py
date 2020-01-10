import os
print("Installing Modules...")
os.system("pip install -r requirements.txt")

import urllib.request
url = "https://pjreddie.com/media/files/yolov3.weights"
location = "datasets/ObjectDetection/yolov3.weights"
print("Downloading File...")
urllib.request.urlretrieve(url, location)
print("File Downloaded")