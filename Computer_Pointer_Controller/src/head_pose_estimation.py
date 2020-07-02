'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
from openvino.inference_engine import IENetwork, IECore
import warnings
warnings.filterwarnings("ignore")


class HeadPoseEstimationClass:
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
        unsupported_layers = [R for R in self.model.layers.keys() if R not in supported_layers]

        if len(unsupported_layers) != 0:
            print("Unsupported layers found ... Adding specified extension ...")
            
            self.core.add_extension(self.extension, self.device)
        
        # Load model
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        

    def predict(self, image):
        
        self.pre_image = self.preprocess_input(image)
        self.results   = self.net.infer(inputs={self.input_name: self.pre_image})
        self.output_angles = self.preprocess_output(self.results)
        
        return self.output_angles
    

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

    def preprocess_output(self, outputs):
        
        output = []
        
        # Get head position angles
        output.append(outputs['angle_y_fc'].tolist()[0][0])
        output.append(outputs['angle_p_fc'].tolist()[0][0])
        output.append(outputs['angle_r_fc'].tolist()[0][0])
        
        return output
