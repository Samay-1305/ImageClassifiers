from modules.ImageClassifiers import *

IMAGE_FILE = 'images/Image.jpg'
CLASSIFIED_IMAGE_FILE = 'images/Image-Clf.jpg'

if __name__ == '__main__':
    image_processor = ImageMultiProcessing('datasets/common/')
    object_detector = ObjectDetection('datasets/ObjectDetection/')
    image = image_processor.load_image(IMAGE_FILE)
    object_detections = object_detector.classify(image)
    drawn_image = image_processor.show_classification(image, object_detections)
    image_processor.save_image(drawn_image, CLASSIFIED_IMAGE_FILE)
    image_processor.show_image(drawn_image)
