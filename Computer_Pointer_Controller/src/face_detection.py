'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
from openvino.inference_engine import IENetwork, IECore
import warnings
warnings.filterwarnings("ignore")

class FaceDetectionClass:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold, extensions=None):
        
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + ".xml"
        self.device    = device
        self.threshold = threshold
        self.extension = extensions
        self.cropped_face_image = None
        self.first_face_coordinates = None
        self.faces_coordinates = None
        self.results = None
        self.p_image = None
        self.net = None

        self.check_model(self.model_structure, self.model_weights)

        self.input_name   = next(iter(self.model.inputs))
        self.input_shape  = self.model.inputs[self.input_name].shape
        self.output_name  = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        

    def load_model(self):
        
        self.core = IECore()

        supported_layers   = self.core.query_network(self.model, self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            print("Unsupported layers found ... Adding specified extension ...")
            
            self.core.add_extension(self.extension, self.device)
            supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        
        # Load model
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        

    def predict(self, image):
        
        self.p_image = self.preprocess_input(image)
        self.results = self.net.infer({self.input_name: self.p_image})
        self.faces_coordinates = self.preprocess_output(self.results, image)

        if len(self.faces_coordinates) == 0:
            print("No Face in current frame, Next frame will be processed..")
            return 0, 0
        
        # Take only the first detected face, if multiple faces detected
        self.first_face_coordinates = self.faces_coordinates[0]
        cropped_face_image = image[self.first_face_coordinates[1]:self.first_face_coordinates[3],
                             self.first_face_coordinates[0]:self.first_face_coordinates[2]]

        return self.first_face_coordinates, cropped_face_image
    

    def check_model(self, model_structure, model_weights):
        try:
            self.model=IENetwork(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        

    def preprocess_input(self, image):
        
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame
    

    def preprocess_output(self, outputs, image):
        
        faces_coordinates = []
        outs = outputs[self.output_name][0][0]
        
        for box in outs:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                faces_coordinates.append([xmin, ymin, xmax, ymax])
        
        return faces_coordinates
