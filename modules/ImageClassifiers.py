from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths
import numpy as np
import imutils
import pickle
import json
import cv2
import os

"""
Custom Built Module to work with Images for object and face detections.

Methods:
ImageMultiProcessing: Handles basic image processing to work with data for classification.
ObjectDetection: Handles loading the dataset and classifying of objects present in an image.
FaceDetection: Handles creation, loading of the dataset and classification of faces.
"""

class ImageMultiProcessing:
    def __init__(self, directory, distance_file='distance_data.json'):
        """Initialise Files"""
        current_directory = os.getcwd()
        self.directory = os.path.join(current_directory, directory) if directory != '' else current_directory
        self.distance_file_path = os.path.join(self.directory, distance_file)
        with open(self.distance_file_path, 'r') as file_object:
            self.distance_data = json.loads(file_object.read())
        self.stream = None

    def initialise_webcam(self):
        """Initialise Webcam and connect to it"""
        self.stream = cv2.VideoCapture(0)

    def initialise_video(self, filepath=''):
        """Load a Video using cv2 for classification purposes"""
        self.stream = cv2.VideoCapture(filepath)

    def get_current_frame(self):
        """Get image from Video or Webcam feed"""
        if self.stream is None:
            self.initialise_webcam()
        ret, frame = self.stream.read()
        return frame

    def load_image(self, filepath='images/Image.jpg'):
        """Convert image to image object for classification"""
        image_object = cv2.imread(filepath)
        return image_object

    def save_image(self, image_object, filepath='images/Image-Save.jpg'):
        """Convert image object back to image to save as a file"""
        cv2.imwrite(filepath, image_object)

    def show_image(self, image_object, window_name='Image', timeout=0):
        """Display unclassified/classified result from image object"""
        cv2.imshow(window_name, image_object)
        cv2.waitKey(timeout)
    
    def get_image_dimensions(self, image_object):
        """Get the width and height of an image_object for calculations"""
        (image_height, image_width) = image_object.shape[:2]
        return [image_width, image_height]

    def compute_position(self, classification_data, image_dimensions=[300, 300]):
        """Update classified data with approximate distance and position in surroundings"""
        for ind, detection in enumerate(classification_data):
            distance = None
            X, Y, W, H = detection["bounding_box"]
            rel_x = X + (W/2)
            if rel_x < img_width*(30/100):
                pos_text = "towards your left"
            elif rel_x <= img_width*(45/100):
                pos_text = "slighly to your left"
            elif rel_x < img_width*(55/100):
                pos_text = "ahead"
            elif rel_x <= img_width*(70/100):
                pos_text = "slightly to your right"
            else:
                pos_text = "towards your right"
            if detection['label'].lower() in self.distance_data.keys():
                known_width, known_distance = self.distance_data[detection['label']]
                distance = round(((known_width * known_distance) / w), 2)
            classification_data[ind]["distance"] = distance
            classification_data[ind]["position"] = pos_text
        return classification_data

    def assign_name(self, object_detection_data=[], face_detection_data=[]):
        """Replace classified object detections title with individual names if trained"""
        for ind, detection in enumerate(object_detection_data):
            X, Y, W, H = detection["bounding_box"]
            if detection["label"].lower() == 'person':
                for face_data in face_detection_data:
                    x, y, w, h = face_data["bounding_box"]
                    center = [x+(w/2), y+(h/2)]
                    if x in range(X, X+W+1) and y in range(Y, Y+H+1):
                        object_detection_data[ind]["label"] = str(face_data["label"])
                        break
        return object_detection_data

    def show_classification(self, image_object, classification_data=[]):
        """Draw the required boxes and labels on the image"""
        for classification in classification_data:
            X, Y, W, H = classification["bounding_box"]
            color = classification["color"]
            confidence = classification["confidence"]
            cv2.rectangle(image_object, (X, Y), (X+W, Y+H), color, 2)
            image_text = "{}: {}".format(classification["label"], confidence)
            cv2.putText(image_object, image_text, (X, Y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image_object

class ObjectDetection:
    def __init__(self, directory='', labels_file='coco.names', config_file='yolov3.cfg', weights_file='yolov3.weights', confidence_threshold=50):
        """Initialise the files for classifying objects"""
        current_directory = os.getcwd()
        self.directory = os.path.join(current_directory, directory) if directory != '' else current_directory
        self.confidence_threshold = float(confidence_threshold)/100.0
        self.threshold = 0.30
        self.labels_file_path = os.path.join(directory, labels_file)
        self.config_file_path = os.path.join(directory, config_file)
        self.weights_file_path = os.path.join(directory, weights_file)
        np.random.seed(42)
        with open(self.labels_file_path, 'r') as file_object:
            self.detection_labels = file_object.read().strip().split("\n")
        self.detection_colors = np.random.randint(0, 255, size=(len(self.detection_labels), 3),dtype="uint8")
        self.neural_net = cv2.dnn.readNetFromDarknet(self.config_file_path, self.weights_file_path)

    def classify(self, image_object):
        """To get objects and classified image data from an image object"""
        (image_height, image_width) = image_object.shape[:2]
        layer_names = self.neural_net.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in self.neural_net.getUnconnectedOutLayers()]
        image_blob = cv2.dnn.blobFromImage(image_object, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.neural_net.setInput(image_blob)
        layer_outputs = self.neural_net.forward(layer_names)
        classifications = {
            "bounding_boxes": [],
            "confidences": [],
            "class_ids": [],
        }
        for layer in layer_outputs:
            for detection in layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= self.confidence_threshold:
                    bounding_box = detection[:4] * np.array([image_width, image_height, image_width, image_height])
                    (center_x, center_y, width, height) = bounding_box.astype("int")
                    X = int(center_x - (width/2))
                    Y = int(center_y - (height/2))
                    classifications["bounding_boxes"].append([X, Y, int(width), int(height)])
                    classifications["confidences"].append(float(confidence))
                    classifications["class_ids"].append(class_id)
        classification_indexes = cv2.dnn.NMSBoxes(
            classifications["bounding_boxes"], 
            classifications["confidences"], 
            self.confidence_threshold, 
            self.threshold
            )
        detections = []
        if len(classification_indexes) > 0:
            for index in classification_indexes.flatten():
                (X, Y, W, H) = classifications["bounding_boxes"][index]
                confidence = round(classifications["confidences"][index], 2)
                class_id = classifications["class_ids"][index]
                color = [int(c_val) for c_val in self.detection_colors[class_id]]
                detection_result = {
                    "label": self.detection_labels[class_id],
                    "confidence": confidence,
                    "bounding_box": [X, Y, W, H],
                    "color": color,
                }
                detections.append(detection_result)
        return detections

class FaceDetection:
    def __init__(self, directory='', proto_file='deploy.prototxt', model_file='res_ssd_300x300.caffemodel', embedd_file='openface_nn4.small2.v1.t7', output_file='embeddings.pickle', recog_file='recognizer.pickle', encoder_file='encoder.pickle', confidence_threshold=50):
        """Initialise the files for classifying objects"""
        current_directory = os.getcwd()
        self.directory = os.path.join(current_directory, directory) if directory != '' else current_directory
        self.confidence_threshold = float(confidence_threshold)/100.0
        self.proto_file_path = os.path.join(self.directory, proto_file)
        self.model_file_path = os.path.join(self.directory, model_file)
        self.embedd_file_path = os.path.join(self.directory, embedd_file)
        self.output_file_path = os.path.join(self.directory, output_file)
        self.recog_file_path = os.path.join(self.directory, recog_file)
        self.encoder_file_path = os.path.join(self.directory, encoder_file)
        self.face_detector = cv2.dnn.readNetFromCaffe(self.proto_file_path, self.model_file_path)
        self.data_embedder = cv2.dnn.readNetFromTorch(self.embedd_file_path)
        self.image_recognizer = pickle.loads(open(self.recog_file_path, 'rb').read())
        self.label_encoder = pickle.loads(open(self.encoder_file_path, 'rb').read())

    def train_from_images(self, images_sub_directory='images/'):
        """Train the face detection dataset and save the changes"""
        images_directory = os.path.join(self.directory, images_sub_directory)
        image_files = list(paths.list_images(images_directory))
        training_data = []
        training_labels = []
        instances = 0
        for (i, image_path) in enumerate(image_files):
            image_label = image_path.replace(r"\\", "/").replace("\\", "/").split("/")[-2]
            image_file = imutils.resize(cv2.imread(image_path), width=600)
            (h, w) = image_file.shape[:2]
            image_blob = cv2.dnn.blobFromImage(cv2.resize(image_file, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            self.face_detector.setInput(image_blob)
            image_detections = self.face_detector.forward()
            if len(image_detections) > 0:
                i = np.argmax(image_detections[0, 0, :, 2])
                confidence = image_detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    box_data = image_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x0, y0, x1, y1 = box_data.astype('int')
                    face_image = image_file[y0:y1, x0:x1]
                    if min(list(face_image.shape[:2])) < 20:
                        continue
                    face_image_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    self.data_embedder.setInput(face_image_blob)
                    visual_encoder = self.data_embedder.forward()
                    image_data = visual_encoder.flatten()
                    training_data.append(image_data)
                    training_labels.append(image_label)
                    instances += 1
        embedding_data = {
            "data": training_data,
            "labels": training_labels
        }
        with open(self.output_file_path, 'wb') as file_object:
            file_object.write(pickle.dumps(embedding_data))
        self.label_encoder = LabelEncoder()
        train_data = embedding_data['data']
        train_labels = self.label_encoder.fit_transform(embedding_data['labels'])
        clf = SVC(C=1.0, kernel="linear", probability=True)
        clf.fit(train_data, train_labels)
        with open(self.recog_file_path, 'wb') as file_object:
            file_object.write(pickle.dumps(clf))
        with open(self.encoder_file_path, 'wb') as file_object:
            file_object.write(pickle.dumps(self.label_encoder))

    def classify(self, image_object):
        """To get faces and classified image data from an image object"""
        required_confidence = 0.5
        (h, w) = image_object.shape[:2]
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image_object, (300, 300),), 1.0, (300,300), (104.0, 117.0, 123.0), swapRB=False, crop=False)
        self.face_detector.setInput(image_blob)
        image_detections = self.face_detector.forward()
        detections = []
        for i in range(image_detections.shape[2]):
            confidence = round(image_detections[0, 0, i, 2], 2)
            if confidence >= self.confidence_threshold:
                box_data = image_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x0, y0, x1, y1 = box_data.astype('int')
                face_image = image_object[y0:y1, x0:x1]
                if min(face_image.shape[:2]) < 20:
                    continue
                face_image_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.data_embedder.setInput(face_image_blob)
                visual_encoder = self.data_embedder.forward()
                prediction = self.image_recognizer.predict_proba(visual_encoder)[0]
                name = self.label_encoder.classes_[np.argmax(prediction)]
                detection_result = {
                    "label": name,
                    "confidence": confidence,
                    "bounding_box": [x0, y0, x1-x0, y1-y0],
                    "color": (255, 255, 255),
                }
                detections.append(detection_result)
        return detections
