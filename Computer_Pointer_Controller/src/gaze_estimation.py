'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
from openvino.inference_engine import IENetwork, IECore
import warnings
import math
warnings.filterwarnings("ignore")


class GazeEstimationClass:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_weights   = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions

        self.check_model(self.model_structure, self.model_weights)

        self.input_name = next(iter(self.model.inputs))
        
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
        

    def predict(self, left_eye, right_eye, head_pose_out):
        
        self.left_eye_image, self.right_eye_image = self.preprocess_input(left_eye, right_eye)
        self.results = self.net.infer(inputs={'left_eye_image': self.left_eye_image, 'right_eye_image': self.right_eye_image,
                                              'head_pose_angles': head_pose_out})
        
        self.mouse_coordinate = self.preprocess_output(self.results, head_pose_out)

        return self.mouse_coordinate
        

    def check_model(self, model_structure, model_weights):
        try:
            self.model=IENetwork(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
    

    def preprocess_input(self, left_eye, right_eye):
        
        left_eye_image = cv2.resize(left_eye, (60, 60))
        left_eye_image = left_eye_image.transpose((2, 0, 1))
        left_eye_image = left_eye_image.reshape(1, *left_eye_image.shape)

        right_eye_image = cv2.resize(right_eye, (60, 60))
        right_eye_image = right_eye_image.transpose((2, 0, 1))
        right_eye_image = right_eye_image.reshape(1, *right_eye_image.shape)

        return left_eye_image, right_eye_image
    
    
    def preprocess_output(self, outputs, head_pose_out):
        
        outputs    = outputs[self.output_name][0]
        z = head_pose_out[2]
        cos_theta  = math.cos(z * math.pi / 180)
        sin_theta  = math.sin(z * math.pi / 180)

        x_value = outputs[0] * cos_theta + outputs[1] * sin_theta
        y_value = outputs[1] * cos_theta - outputs[0] * sin_theta

        return (x_value, y_value)
