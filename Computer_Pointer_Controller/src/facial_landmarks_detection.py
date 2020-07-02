'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
from openvino.inference_engine import IENetwork, IECore
import warnings
warnings.filterwarnings("ignore")


class FacialLandmarksClass:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_weights   = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device    = device
        self.extension = extensions

        self.check_model(self.model_structure, self.model_weights)

        self.input_name   = next(iter(self.model.inputs))
        self.input_shape  = self.model.inputs[self.input_name].shape
        self.output_name  = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        

    def load_model(self):
        
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.core  = IECore()
        supported_layers   = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            print("Unsupported layers found ... Adding specified extension ...")
            
            self.core.add_extension(self.extension, self.device)
        
        # Load model
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        

    def predict(self, image):
        
        self.p_image = self.preprocess_input(image)
        self.results = self.net.infer(inputs={self.input_name: self.p_image})
        self.output  = self.preprocess_output(self.results, image)
        
        # Process output and send cropped image, coordinates
        left_eye_x_min = self.output['left_eye_x_coordinate'] - 15
        left_eye_x_max = self.output['left_eye_x_coordinate'] + 15
        left_eye_y_min = self.output['left_eye_y_coordinate'] - 15
        left_eye_y_max = self.output['left_eye_y_coordinate'] + 15

        right_eye_x_min = self.output['right_eye_x_coordinate'] - 15
        right_eye_x_max = self.output['right_eye_x_coordinate'] + 15
        right_eye_y_min = self.output['right_eye_y_coordinate'] - 15
        right_eye_y_max = self.output['right_eye_y_coordinate'] + 15

        self.eye_coord = [[left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max],
                          [right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max]]
        
        left_eye_image  = image[left_eye_x_min:left_eye_x_max, left_eye_y_min:left_eye_y_max]
        right_eye_image = image[right_eye_x_min:right_eye_x_max, right_eye_y_min:right_eye_y_max]

        return left_eye_image, right_eye_image, self.eye_coord
    

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
        
        outputs = outputs[self.output_name][0]
        w, h = image.shape[1], image.shape[0]
        
        # Coordinates of eye
        left_x  = int(outputs[0] * w)
        left_y  = int(outputs[1] * h)
        right_x = int(outputs[2] * w)
        right_y = int(outputs[3] * h)

        return {'left_eye_x_coordinate': left_x,   'left_eye_y_coordinate': left_y,
                'right_eye_x_coordinate': right_x, 'right_eye_y_coordinate': right_y}
