# ImageClassifiers
A custom built module based on existing Object and Face Detection algorithms using python and opencv to  perform simply classifications on images. 

# Prerequisites
Run setup.py to complete the setup and verification of modules and files
Create a folder for each person with all images of them in the directory

# ImageMultiProcessor
Functions used to help in classification and working with image objects

* ImageMultiProcessor(directory, distance_file)
* initialise_webcam(): None -> Start a stream from the attached webcam
* initialise_video(filepath): None -> Start a stream from a video file
* get_current_frame(): Image Object -> Get an Image Object from the current stream
* load_image(filepath): Image Object -> Get an Image Object from an Image file
* save_image(image_object, filepath): None -> Save an Image Object to a file
* show_image(image_object, window_name, timeout): None -> Show an Image Object
* get_image_dimensions(image_object): List -> Get the Width and Height of an image from an Image Object
* compute_position(classification_data, image_dimensions): List -> Get Distance and position of classified objects
* assign_name(object_detection_data, face_detection_data): List -> Assign a name to persons from classified dat
* show_classification(image_object, classification_data): Image Object -> Creates an Image Object that consists of an overlay with the classification_data

# ObjectDetection
Functions used for classifying objects in image objects

* ObjectDetection(directory='', labels_file='coco.names', config_file='yolov3.cfg', weights_file='yolov3.weights', confidence_threshold=50)
* classify(image_object): List -> Classify all the objects in an image

# FaceDetection
Functions used for classifying and training faces in image objects

* FaceDetection(directory='', proto_file='deploy.prototxt', model_file='res_ssd_300x300.caffemodel', embedd_file='openface_nn4.small2.v1.t7', output_file='embeddings.pickle', recog_file='recognizer.pickle', encoder_file='encoder.pickle', confidence_threshold=50)
* train_from_images(images_sub_directory='images/'): None -> Creates a custom dataset based on supplied images
* classify(image_object): List -> Classify all the Faces in an image
