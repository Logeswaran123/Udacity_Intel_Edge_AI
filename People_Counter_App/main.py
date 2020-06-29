"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60



def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def draw_boxes(frame, result, args, width, height):   
    count = 0     
    classid = 0
    for box in result[0][0]:  
        confidence = box[2]  
        # if id == 1, it is person 
        classid = int(box[1]) 
        if classid == 1: 
            if confidence >= args.prob_threshold: 
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)  
                # draw bbox
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1) 
                Person_confidence = '%s: %.1f%%' % ("Person", round(confidence * 100, 1))
                cv2.putText(frame, Person_confidence, (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                count += 1    
                
    return frame, count

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Checks for live feed camera
    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)

    w = int(cap.get(3))
    h = int(cap.get(4))

    in_shape = network_shape['image_tensor']

    #iniatilize variables    
    counter = 0
    duration = 0
    total_count = 0
    current_count = 0 
    total_inference_time = 0
    last_count = 0 
    threshold_value = 2 
    request_id = 0
    
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60) 

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
  

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
        duration_report = None
        infer_network.exec_net(image_p, request_id)
        
        # Calculating time for the performance in different models
        infer_timer = time.time()
        
        ### TODO: Wait for the result ### 
        if infer_network.wait() == 0:
            # Calculating time for the performance in different models
            inferece_time = time.time() - infer_timer  
            
            ### TODO: Get the results of the inference request ###  
            result = infer_network.get_output() 
            
            ### TODO: Extract any desired stats from the results ### 
            frame, count = draw_boxes(frame, result, args, w, h)   
            
            ### TODO: Calculate and send relevant information on ###   
            current_count = count    
            if current_count > last_count:
                # if someone enter frame, time start counting
                start_time = time.time() 
                total_count = total_count + current_count - last_count 
                
            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # if detection failed and double counted, decrease its value
                if duration < threshold_value:
                    total_count = total_count - 1  
                if duration >= 4:
                    ### Topic "person/duration": key of "duration" ###
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    
                    ### Topic "person": keys of "count" and "total" ###
                    client.publish("person", json.dumps({"total": total_count})) 
            
            ### Current Count, Total Count and Duration to the MQTT server ###
            client.publish("person", json.dumps({"count": count}))
            last_count = current_count
             
            if key_pressed == 27:
                break 
 

        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()



def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
