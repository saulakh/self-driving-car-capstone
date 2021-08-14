from styx_msgs.msg import TrafficLight
import keras
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        # Load keras model for traffic lights
        #model = keras.models.load_model('lights_model.h5')
        #print("Model loaded")
        
        return

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        """
        #TODO implement light color prediction
        np_image = np.asarray(image)
        
        # Best traffic light prediction from softmax probabilities
        softmax = self.model.predict(np_image)
        pred = np.argmax(softmax, axis=1)

        # Return predicted traffic light color
        labels = ['red', 'yellow', 'green']
        
        if (pred == 0):
            return TrafficLight.RED
        elif (pred == 1):
            return TrafficLight.YELLOW
        elif (pred == 2):
            return TrafficLight.GREEN
        """
        
        return TrafficLight.UNKNOWN
