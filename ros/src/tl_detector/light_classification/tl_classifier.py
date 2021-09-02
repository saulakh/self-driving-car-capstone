from styx_msgs.msg import TrafficLight
import keras
import numpy as np
import tensorflow as tf
import cv2
import object_detection

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        # Path for COCO SSD model graph
        self.graph_file = 'light_classification/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

        # Load graph file
        def load_graph(graph_file):
            """Loads a frozen inference graph"""
            graph = tf.Graph()
            with graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(graph_file, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            return graph

        # Load graph file
        self.detection_graph = load_graph(self.graph_file)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        print("Graph loaded")
        
        # Load keras model for traffic lights
        #self.model = keras.models.load_model('light_classification/lights_model.h5')
        #print("Model loaded")
        
        return

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        #TODO implement light color prediction
        
        # Get cropped image of traffic light
        light_image = object_detection.get_traffic_light(image, self.detection_graph, self.image_tensor, self.detection_boxes, self.detection_scores, self.detection_classes)
        light_exp = np.expand_dims(np.asarray(light_image), axis=0)
        
        """
        # Best traffic light prediction from softmax probabilities
        softmax = self.model.predict(light_exp)
        pred = np.argmax(softmax, axis=1)

        # Return predicted traffic light color
        labels = ['red', 'yellow', 'green']
        
        if (pred[0] == 0):
            return TrafficLight.RED
        elif (pred[0] == 1):
            return TrafficLight.YELLOW
        elif (pred[0] == 2):
            return TrafficLight.GREEN
        """
        
        # Find brightest light from HSV color space
        light_hsv = cv2.cvtColor(np.asarray(light_image), cv2.COLOR_BGR2HSV)[:,:,-1]
        height = light_hsv.shape[0]
        y,x = np.where(light_hsv >= 0.8*light_hsv.max())
        light_ratio = round(y.mean()/height, 2)
        print("Light ratio:", light_ratio)

        # Predict light color from location
        if light_ratio <= 0.45:
            return TrafficLight.RED
        if 0.45 < light_ratio <= 0.6:
            return TrafficLight.YELLOW
        if light_ratio > 0.6:
            return TrafficLight.GREEN
        
        return TrafficLight.UNKNOWN
