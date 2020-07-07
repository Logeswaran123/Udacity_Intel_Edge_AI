# Gaze Estimation

import cv2
from openvino.inference_engine import IENetwork, IECore
import math



class GazeEstimationClass:
    
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_weights   = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
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
        self.output_name  = next(iter(self.network.outputs))
        

    def predict(self, left_eye, right_eye, head_pose_out):
        
        p_left_eye, p_right_eye = self.preprocess_input(left_eye, right_eye)
        results = self.exec_net.infer({'head_pose_angles': head_pose_out, 'left_eye_image':p_left_eye, 'right_eye_image': p_right_eye})
        mouse_coordinate, gaze_vector = self.preprocess_output(results, head_pose_out)

        return mouse_coordinate, gaze_vector
        

    def check_model(self, model_structure, model_weights):
        try:
            self.network = IENetwork(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialize Network")
    

    def preprocess_input(self, left_eye, right_eye):
        # model requires shape [1x3x60x60]
        p_left_eye = cv2.resize(left_eye, (60, 60))
        p_left_eye = p_left_eye.transpose((2, 0, 1))
        p_left_eye = p_left_eye.reshape(1, *p_left_eye.shape)

        p_right_eye = cv2.resize(right_eye, (60, 60))
        p_right_eye = p_right_eye.transpose((2, 0, 1))
        p_right_eye = p_right_eye.reshape(1, *p_right_eye.shape)

        return p_left_eye, p_right_eye
    
    
    def preprocess_output(self, outputs, head_pose_out):
        
        # Calculate x,y for mouse movement
        gaze_vector = outputs[self.output_name][0]
        angle_r_fc  = head_pose_out[2]
        cosine  = math.cos(angle_r_fc * math.pi / 180)
        sine  = math.sin(angle_r_fc * math.pi / 180)

        x_value = gaze_vector[0] * cosine + gaze_vector[1] * sine
        y_value = - gaze_vector[0] * sine + gaze_vector[1] * cosine

        return (x_value, y_value), gaze_vector
    
    
def all_layers_supported(engine, network):
        
    layers_supported = engine.query_network(network, device_name='CPU')
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False

    return all_supported
