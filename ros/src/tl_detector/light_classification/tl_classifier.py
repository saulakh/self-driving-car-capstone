from styx_msgs.msg import TrafficLight
import keras
import numpy as np
import tensorflow as tf
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        # Path for COCO SSD model graph
        self.graph_file = 'light_classification/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

        # Load graph file
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            # Create tf session
            self.sess = tf.Session(graph=self.detection_graph)

        # Load graph file
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        print("Graph loaded")
        
        return

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        #TODO implement light color prediction
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        
        def filter_boxes(min_score, boxes, scores, classes):
            """Return boxes with a confidence >= `min_score`"""
            n = len(classes)
            idxs = []
            for i in range(n):
                if scores[i] >= min_score:
                    idxs.append(i)
    
            filtered_boxes = boxes[idxs, ...]
            filtered_scores = scores[idxs, ...]
            filtered_classes = classes[idxs, ...]
            return filtered_boxes, filtered_scores, filtered_classes

        def to_image_coords(boxes, height, width):
            """
            The original box coordinate output is normalized, i.e [0, 1].
    
            This converts it back to the original coordinate based on the image
            size.
            """
            box_coords = np.zeros_like(boxes)
            box_coords[:, 0] = boxes[:, 0] * height
            box_coords[:, 1] = boxes[:, 1] * width
            box_coords[:, 2] = boxes[:, 2] * height
            box_coords[:, 3] = boxes[:, 3] * width
    
            return box_coords
        
        with self.detection_graph.as_default():
            # Actual detection.
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                            feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.8
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            height, width, channels = image.shape
            box_coords = to_image_coords(boxes, height, width)
            
            #print("Traffic lights found:", len(box_coords))
            
            if len(box_coords) > 0:
                          
                # Use box coords to crop traffic light image
                img = np.array(image)
                light_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get box coordinates for original image size 
                box_coord = box_coords[0]
                bot = int(box_coord[0])
                left = int(box_coord[1])
                top = int(box_coord[2])
                right = int(box_coord[3])

                # Crop image within bounding box
                light_img = light_rgb[bot:top, left:right]
                light_output = cv2.resize(light_img, (48,108), interpolation = cv2.INTER_AREA)
        
                # Find brightest light from V-channel of HSV color space
                light_hsv = cv2.cvtColor(np.asarray(light_output), cv2.COLOR_BGR2HSV)[:,:,-1]
                cv2.imwrite("light_hsv.jpg", light_hsv)
                height = light_hsv.shape[0]
                y,x = np.where(light_hsv >= 0.8*light_hsv.max())
                light_ratio = round(y.mean()/height, 2)
                print("Light ratio:", light_ratio)

                # Predict light color from light ratio
                if light_ratio <= 0.5:
                    return TrafficLight.RED
                if 0.5 < light_ratio <= 0.6:
                    return TrafficLight.YELLOW
                if light_ratio > 0.6:
                    return TrafficLight.GREEN
                
            else:
                return TrafficLight.UNKNOWN
