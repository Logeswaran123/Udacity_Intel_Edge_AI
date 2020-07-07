# Facial Landmarks Detection

import cv2
from openvino.inference_engine import IENetwork, IECore

class FacialLandmarksClass:
    
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_weights   = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device    = device
        self.extension = extensions
        self.network = None
        self.plugin = None
        self.exec_net = None


    def load_model(self):
        
        self.plugin = IECore()
        self.check_model(self.model_structure, self.model_weights)

        if not all_layers_supported(self.plugin, self.network):
            self.plugin.add_extension(self.extension, self.device)
        
        # Load model
        self.exec_net = self.plugin.load_network(network = self.network, device_name= self.device, num_requests=1)
        self.input_name   = next(iter(self.network.inputs))
        self.input_shape  = self.network.inputs[self.input_name].shape
        self.output_name  = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape
        

    def predict(self, image):
        
        p_image = self.preprocess_input(image)
        results = self.exec_net.infer(inputs={self.input_name: p_image})
        output  = self.preprocess_output(results, image)
        
        # Process output
        left_x_min = output[0] - 15
        left_x_max = output[0] + 15
        left_y_min = output[1] - 15
        left_y_max = output[1] + 15

        right_x_min = output[2] - 15
        right_x_max = output[2] + 15
        right_y_min = output[3] - 15
        right_y_max = output[3] + 15

        coordinates = [[left_x_min, left_y_min, left_x_max, left_y_max],
                       [right_x_min, right_y_min, right_x_max, right_y_max]]
        
    
        left_eye  = image[left_y_min :left_y_max,  left_x_min :left_x_max]
        right_eye = image[right_y_min:right_y_max, right_x_min:right_x_max]

        return left_eye, right_eye, coordinates
    

    def check_model(self, model_structure, model_weights):
        try:
            self.network = IENetwork(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialize Network")


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

        return (left_x, left_y, right_x, right_y)


def all_layers_supported(engine, network):
        
    layers_supported = engine.query_network(network, device_name='CPU')
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False

    return all_supported
