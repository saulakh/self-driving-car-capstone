from styx_msgs.msg import TrafficLight
import keras
import numpy as np
import object_detection

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        # Path for COCO SSD model graph
        COCO_GRAPH = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

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
        self.detection_graph = load_graph(COCO_GRAPH)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        
        # Load keras model for traffic lights
        self.model = keras.models.load_model('lights_model.h5')
        print("Model loaded")
        
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
        light_image = np.expand_dims(np.asarray(light_image), axis=0)
        
        # Best traffic light prediction from softmax probabilities
        softmax = self.model.predict(light_image)
        pred = np.argmax(softmax, axis=1)

        # Return predicted traffic light color
        labels = ['red', 'yellow', 'green']
        
        if (pred[0] == 0):
            return TrafficLight.RED
        elif (pred[0] == 1):
            return TrafficLight.YELLOW
        elif (pred[0] == 2):
            return TrafficLight.GREEN
        
        return TrafficLight.UNKNOWN
