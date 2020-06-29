# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...
Ans:
Depending upon the framework used, converting custom layers invloves different steps.
For Tensorflow,
1. Register the Custom layer as an extension in model optimizer.
2. If you have subgraph that cannot be represented in Intermediate representation. Then, tensorflow provides an option to include that for the operation.

For MXNet,
This process is similar to Tensorflow custom layer conversion.

For Caffe,
Register the custom layers as extensions to the Model Optimizer and generate valid Intermediate Representation(IR).

Some of the potential reasons for handling custom layers are...
Ans:
The Intel distribution of OpenVino toolkit supports layers from various frameworks like TensorFlow, Caffe, MXNet, Kaldi and ONYX.

The list of known layers varies with frameworks. If any layer is not in list of knpwn layers, then Model Optimizer classifies it as custom.

The model optimizer extracts information from the input model. Then, identifies the custom layers. It does model optimization by quantization, freezing and fusing layers. Finally, intermediate representation is generated from the model.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was not much as the SSD_MobileNet_V2 was missing detections of not moving persons in frame.

The size of the model pre-coversion was 66.4MB and post-conversion was 64.1MB.

The inference time of the model post-conversion was arount 70-90ms and pre-conversion was around 2500ms.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are,
1. Detect the number of people in a room and have it below a particular count.
2. Queueing system to check for free space in a queue.
3. Automated security system

Each of these use cases would be useful because,
1. No individual is required to do the monitoring job.
2. All people in the frame can be captured and the information can be stored in a database.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows,

1. Lighting of the space can have significant effects on model detetction accuracy. Poorly lit places can cause the model to not detect the people in the frame. It might produce more false positives and false negatives.
2. Camera angle and focal length must be pre-adjusted while setting up the system. Also, the image size must be pre-processed to be compatible with the model input.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Faster-RCNN v2 model]
  - [Model Source]
  ```
  wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  ```
  ```
  tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  ```
  - I converted the model to an Intermediate Representation with the following arguments...
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
  ```
  - The model was insufficient for the app because...
  I faced issue with unsupported layers (custom layers) and CPU extension.
  - I tried to improve the model for the app by...
  trying a different model
  
- Model 2: [SSD Mobile Net v2]
  - [Model Source]
  ```
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  ```
  ```
  tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  ```
  - I converted the model to an Intermediate Representation with the following arguments...
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - Run the app by...
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```

