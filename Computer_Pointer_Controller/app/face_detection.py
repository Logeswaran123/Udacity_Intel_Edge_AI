# Face Detection

import cv2
from openvino.inference_engine import IENetwork, IECore


class FaceDetectionClass:
    
    def __init__(self, model_name, device, threshold, extensions=None):
        
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + ".xml"
        self.device    = device
        self.threshold = threshold
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
        results = self.exec_net.infer({self.input_name: p_image})
        coordinates = self.preprocess_output(results, image)
        
        if len(coordinates) == 0:
            return 0, 0
        
        cropped_face = image[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        
        return coordinates, cropped_face
    

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
        
        coordinates = []
        outputs = outputs[self.output_name][0][0]
        w, h = image.shape[1], image.shape[0]
        
        for output in outputs:
            confidence = output[2]
            if confidence >= self.threshold:
                xmin = int(output[3] * w)
                ymin = int(output[4] * h)
                xmax = int(output[5] * w)
                ymax = int(output[6] * h)
                coordinates.append([xmin, ymin, xmax, ymax])
        
        return list(map(int, coordinates[0]))


def all_layers_supported(engine, network):
        
    layers_supported = engine.query_network(network, device_name='CPU')
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False

    return all_supported
