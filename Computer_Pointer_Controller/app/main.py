'''
App
'''

import os
import cv2
import time
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetectionClass
from facial_landmarks_detection import FacialLandmarksClass
from gaze_estimation import GazeEstimationClass
from head_pose_estimation import HeadPoseEstimationClass
from argparse import ArgumentParser



def build_argparser():
    
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



def main():
    
    # Get command line arguments
    args = build_argparser().parse_args()
    input_path = args.input
    visualize_flag = args.visualization_flag
    
    # Check the input type
    if input_path == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_path):
            print("Input file path not valid")
            exit(1)
        input_feeder = InputFeeder("video", input_path)

    

    # Define model objects
    fd = FaceDetectionClass(args.face_detection_model, args.device, args.prob_threshold, args.cpu_extension)

    fl = FacialLandmarksClass(args.facial_landmarks_model, args.device, args.cpu_extension)
    
    hp = HeadPoseEstimationClass(args.head_pose_model, args.device, args.cpu_extension)

    ge = GazeEstimationClass(args.gaze_estimation_model, args.device, args.cpu_extension)
    
    
    # Load the models
    # Face Detection Model load time
    print("**********************************************************************")
    time1 = time.time()
    fd.load_model()
    Face_detection_time = (time.time() - time1) * 1000
    print("Face detection model load time: {:.3f} ms".format(Face_detection_time))
    
    # Facial Landmarks Detection Model load time
    time2 = time.time()
    fl.load_model()
    Landmarks_detection_time = (time.time() - time2) * 1000
    print("Facial landmarks detection model load time: {:.3f} ms".format(Landmarks_detection_time))
    
    # Head Pose Estimation Model load time
    time3 = time.time()
    hp.load_model()
    Headpose_estimation_time = (time.time() - time3) * 1000
    print("Head pose estimation model load time: {:.3f} ms".format(Headpose_estimation_time))
    
    # Gaze Estimation Model load time
    time4 = time.time()
    ge.load_model()
    Gaze_estimation_time = (time.time() - time4) * 1000
    print("Gaze estimation model load time: {:.3f} ms".format(Gaze_estimation_time))
    
    # Total load time
    total_load_time = time.time() - time1
    print("Total load time: {:.3f} ms".format(total_load_time * 1000))
    print("All models loaded successfully")
    print("**********************************************************************")
    
    input_feeder.load_data()
    print("Input feeder loaded")
    
    # Determining Precsion and Speed for mouse controller 
    mouse_controller = MouseController('high', 'fast')

    frame_count = 0
    start_time = time.time()
    print("Start inferencing on input feed ")
    
    # Loop through input feed till break
    for flag, frame in input_feeder.next_batch():
        
        if not flag:
            break
        keyPressed = cv2.waitKey(60)
        frame_count += 1
        #print(frame.shape[0], frame.shape[1])
        
        # Face Detection
        face_coordinates, face_crop = fd.predict(frame)
        
        # No face
        if len(face_coordinates) == 0:
            print("No face detected")
            continue
        
        # Head position detection
        hp_output = hp.predict(face_crop.copy())
        
        # Landmarks detection
        left_eye, right_eye, eye_coordinate = fl.predict(face_crop.copy())
        
        # Gaze detection
        mouse_coordinate, gaze_vector = ge.predict(left_eye, right_eye, hp_output)

        if len(visualize_flag) != 0:
            frame_copy = frame.copy()
            
            if 'fd' in visualize_flag:
                # Face
                cv2.rectangle(frame_copy, (face_coordinates[0], face_coordinates[1]),
                                          (face_coordinates[2], face_coordinates[3]), (0, 255, 150), 2)
            
            if 'fl' in visualize_flag:
                # Left eye
                cv2.rectangle(frame_copy, (face_coordinates[0] + eye_coordinate[0][0], face_coordinates[1] + eye_coordinate[0][1]), 
                                          (face_coordinates[0] + eye_coordinate[0][2], face_coordinates[1] + eye_coordinate[0][3]), (255, 0, 150), 2)
                # Right eye
                cv2.rectangle(frame_copy, (face_coordinates[0] + eye_coordinate[1][0], face_coordinates[1] + eye_coordinate[1][1]), 
                                          (face_coordinates[0] + eye_coordinate[1][2], face_coordinates[1] + eye_coordinate[1][3]), (255, 0, 150), 2)
                
            if 'hp' in visualize_flag:
                # Head Position
                cv2.putText(frame_copy, "Yaw: {:.1f}".format(hp_output[0]),   (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 150, 0), 3)
                cv2.putText(frame_copy, "Pitch: {:.1f}".format(hp_output[1]), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 150, 0), 3)
                cv2.putText(frame_copy, "Roll: {:.1f}".format(hp_output[2]),  (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 150, 0), 3)
                
            if 'ge' in visualize_flag:
                # Gaze
                x, y = gaze_vector[0:2]
                left_eye_center  = (int(face_coordinates[0] + (eye_coordinate[0][0] + eye_coordinate[0][2])/2), int(face_coordinates[1] + (eye_coordinate[0][1] + eye_coordinate[0][3])/2))
                right_eye_center = (int(face_coordinates[0] + (eye_coordinate[1][0] + eye_coordinate[1][2])/2), int(face_coordinates[1] + (eye_coordinate[1][1] + eye_coordinate[1][3])/2))
                frame_copy = cv2.arrowedLine(frame_copy, left_eye_center, (int(left_eye_center[0]+x*200), int(left_eye_center[1]-y*200)), (57,255,20), 2)
                frame_copy = cv2.arrowedLine(frame_copy, right_eye_center, (int(right_eye_center[0]+x*200), int(right_eye_center[1]-y*200)), (57,255,20), 2)
                
        # Resize frame to fit output inside screen
        img_final = cv2.resize(frame_copy, (960, 540))
        cv2.imshow('Visualization', img_final)
        
        # Move mouse pointer with Gaze Vector (x,y)
        mouse_controller.move(mouse_coordinate[0], mouse_coordinate[1])
        
        # 'ESC' to end
        if keyPressed == 27:
            print("Exited")
            break
    
    input_feeder.close()
    cv2.destroyAllWindows()
    
    # Inference
    
    print("**********************************************************************")
    inference_time = round(time.time() - start_time, 1)
    fps = int(frame_count) / inference_time
    print("Total Inference Time: {} seconds".format(inference_time))
    print("FPS: {:.3f} frames/second".format(fps))
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
    
    ###################


if __name__ == '__main__':
    main()
