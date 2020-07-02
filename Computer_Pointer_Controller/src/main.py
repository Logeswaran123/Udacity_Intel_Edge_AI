'''
App
'''

import os
import numpy as np
import cv2
import time
import math
from face_detection import FaceDetectionClass
from facial_landmarks_detection import FacialLandmarksClass
from gaze_estimation import GazeEstimationClass
from head_pose_estimation import HeadPoseEstimationClass
from mouse_controller import MouseController
from input_feeder import InputFeeder
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")


def build_argparser():
    """
    Parse command line arguments.
    return: command line arguments
    """
    
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model",   required=True, type=str, help="Path to a face detection model xml file")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str, help="Path to a facial landmarks detection model xml file")
    parser.add_argument("-hp", "--head_pose_model",        required=True, type=str, help="Path to a head pose estimation model xml file")
    parser.add_argument("-ge", "--gaze_estimation_model",  required=True, type=str, help="Path to a gaze estimation model xml file")
    parser.add_argument("-i",  "--input",                  required=True, type=str, help="Path to image or video or CAM")
    parser.add_argument("-l",  "--cpu_extension",          required=False,type=str, help="targeted custom layers (CPU)", default=None)
    parser.add_argument("-d",  "--device", type=str, default="CPU", help="Specify the target device (CPU, GPU, FPGA, VPU)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5, help="Probability threshold for detections (0.5 by default)")
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+', default=[],
                        help="Visualize different model output on frame"
                             "fd: Face Detection Model,       fl: Facial Landmark Detection Model"
                             "hp: Head Pose Estimation Model, ge: Gaze Estimation Model")
    
    return parser

# Reference: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    '''
    Parameters
    ----------
    frame : image
        input frame (video or cam).
    center_of_face : 
        center face coordinates.
    yaw : TYPE
        head position parameter.
    pitch : 
        head position parameter.
    roll : 
        head position parameter.
    scale : 
        head position parameter.
    focal_length

    Returns
    -------
    frame : image
        frame with axes position.
    '''
    
    yaw   *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll  *= np.pi / 180.0
    
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    
    R_x = np.array([[1,               0,               0],
                    [0, math.cos(pitch),-math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    R_y = np.array([[math.cos(yaw),   0,  -math.sin(yaw)],
                    [0,               1,               0],
                    [math.sin(yaw),   0,   math.cos(yaw)]])
    R_z = np.array([[math.cos(roll), -math.sin(roll),  0],
                    [math.sin(roll),  math.cos(roll),  0],
                    [0,               0,               1]])
    
    #R = np.dot(R_z, np.dot( R_y, R_x ))
    R = R_z @ R_y @ R_x
    
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    
    xaxis  = np.dot(R, xaxis) + o
    yaxis  = np.dot(R, yaxis) + o
    zaxis  = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2  = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2  = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1  = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2  = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    
    return frame


# Create camera matrix
def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    
    return camera_matrix


def main():
    
    # Get command line arguments
    args = build_argparser().parse_args()
    input_file_path = args.input
    visualize_flag = args.visualization_flag
    
    # Check the input type
    if input_file_path == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path):
            print("Input file path not valid")
            exit(1)
        input_feeder = InputFeeder("video", input_file_path)

    

    # Define model objects
    face_detection_model_object = FaceDetectionClass(model_name=args.face_detection_model,
                                                     device=args.device, threshold=args.prob_threshold,
                                                     extensions=args.cpu_extension)

    facial_landmarks_detection_model_object = FacialLandmarksClass(model_name=args.facial_landmarks_model,
                                                                   device=args.device, 
                                                                   extensions=args.cpu_extension)
    
    head_pose_estimation_model_object = HeadPoseEstimationClass(model_name=args.head_pose_model, 
                                                                device=args.device, 
                                                                extensions=args.cpu_extension)

    gaze_estimation_model_object = GazeEstimationClass(model_name=args.gaze_estimation_model, 
                                                       device=args.device, 
                                                       extensions=args.cpu_extension)
    
    
    # Determining Precsion and Speed for mouse controller 
    #mouse_controller = MouseController('low', 'fast')
    mouse_controller_object = MouseController('medium', 'fast')
    
    # Load the models
    # Face Detection Model load time
    print("**********************************************************************")
    start_time = time.time()
    face_detection_model_object.load_model()
    Face_detection_time = (time.time() - start_time) * 1000
    print("Face detection model load time: {:.3f} ms".format(Face_detection_time))
    
    # Facial Landmarks Detection Model load time
    model_2_start = time.time()
    facial_landmarks_detection_model_object.load_model()
    Landmarks_detection_time = (time.time() - model_2_start) * 1000
    print("Facial landmarks detection model load time: {:.3f} ms".format(Landmarks_detection_time))
    
    # Head Pose Estimation Model load time
    model_3_start = time.time()
    head_pose_estimation_model_object.load_model()
    Headpose_estimation_time = (time.time() - model_3_start) * 1000
    print("Head pose estimation model load time: {:.3f} ms".format(Headpose_estimation_time))
    
    # Gaze Estimation Model load time
    model_4_start = time.time()
    gaze_estimation_model_object.load_model()
    Gaze_estimation_time = (time.time() - model_4_start) * 1000
    print("Gaze estimation model load time: {:.3f} ms".format(Gaze_estimation_time))
    
    # Total load time
    total_load_time = time.time() - start_time
    print("Total load time: {:.3f} ms".format(total_load_time * 1000))
    print("All models loaded successfully")
    print("**********************************************************************")
    
    input_feeder.load_data()
    print("Input feeder loaded")

    count = 0
    start_inference_time = time.time()
    print("Start inferencing on input feed ")
    
    # Loop through input feed till break
    for flag, frame in input_feeder.next_batch():
        
        if not flag:
            break
        keyPressed = cv2.waitKey(60)
        count += 1
        #print(frame.shape[0], frame.shape[1])
        
        # Face Detection
        face_coordinates, face_image = face_detection_model_object.predict(frame)
        
        # No face
        if face_coordinates == 0:
            continue
        
        # Head position detection
        hp_output = head_pose_estimation_model_object.predict(face_image)
        
        # Landmarks detection
        left_eye_image, right_eye_image, eye_coord = facial_landmarks_detection_model_object.predict(face_image)
        
        # Gaze detection
        mouse_coordinate = gaze_estimation_model_object.predict(left_eye_image, right_eye_image, hp_output)

        if len(visualize_flag) != 0:
            frame_copy = frame.copy()
            
            if 'fd' in visualize_flag:
                cv2.rectangle(frame_copy, (face_coordinates[0], face_coordinates[1]),
                                          (face_coordinates[2], face_coordinates[3]), (0, 255, 150), 2)
            
            if 'fl' in visualize_flag:
                #frame_copy = face_image.copy()
                cv2.rectangle(frame_copy, (face_coordinates[0] + eye_coord[0][0], face_coordinates[1] + eye_coord[0][1]), 
                                          (face_coordinates[0] + eye_coord[0][2], face_coordinates[1] + eye_coord[0][3]), (255, 0, 150), 2)
                cv2.rectangle(frame_copy, (face_coordinates[0] + eye_coord[1][0], face_coordinates[1] + eye_coord[1][1]), 
                                          (face_coordinates[0] + eye_coord[1][2], face_coordinates[1] + eye_coord[1][3]), (255, 0, 150), 2)
                
            if 'hp' in visualize_flag:
                cv2.putText(frame_copy, "Yaw: {:.1f}".format(hp_output[0]),   (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 150, 0), 3)
                cv2.putText(frame_copy, "Pitch: {:.1f}".format(hp_output[1]), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 150, 0), 3)
                cv2.putText(frame_copy, "Roll: {:.1f}".format(hp_output[2]),  (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 150, 0), 3)
                
            if 'ge' in visualize_flag:
                yaw = hp_output[0]
                pitch = hp_output[1]
                roll = hp_output[2]
                
                focal_length = 950.0
                scale = 50
                
                center_of_face = (face_coordinates[0] + face_image.shape[1] / 2, face_coordinates[1] + face_image.shape[0] / 2, 0)
                
                draw_axes(frame_copy, center_of_face, yaw, pitch, roll, scale, focal_length)
                
        # Resize frame to fit output inside screen
        if len(visualize_flag) != 0:
            img_final = np.hstack((cv2.resize(frame, (960, 540)), cv2.resize(frame_copy, (960, 540))))
        else:
            img_final = cv2.resize(frame, (960, 540))
        cv2.imshow('Visualization', img_final)
        
        # Move mouse pointer with Gaze Vector (x,y)
        mouse_controller_object.move(mouse_coordinate[0], mouse_coordinate[1])
        
        # 'ESC' to end
        if keyPressed == 27:
            print("Exited")
            break
    
    print("**********************************************************************")
    inference_time = round(time.time() - start_inference_time, 1)
    fps = int(count) / inference_time
    print("Total inference time {} seconds".format(inference_time))
    print("FPS {} frame/second".format(fps))
    print("Input feed ended")
    print("**********************************************************************")
    
    # Write the performance results to txt file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.txt'), 'w') as f:
        f.write("Inference Time: {} ms".format(inference_time) + '\n')
        f.write("FPS: {}".format(fps) + '\n')
        f.write("All model loading times" + '\n')
        f.write("Face Detection model load time: {:.3f} ms".format(Face_detection_time) + '\n')
        f.write("Facial Landmarks model load time: {:.3f} ms".format(Landmarks_detection_time) + '\n')
        f.write("Head Pose Estimation model load time: {:.3f} ms".format(Headpose_estimation_time) + '\n')
        f.write("Gaze Estimation model load time: {:.3f} ms".format(Gaze_estimation_time) + '\n')
        f.write("Total: {:.3f} ms".format(total_load_time * 1000) + '\n')

    input_feeder.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()