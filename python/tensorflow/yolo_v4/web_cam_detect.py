import cv2
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np

import core.utils as utils

from tensorflow.python.saved_model import tag_constants

from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')


# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def loadModelByOpencv( pb_path ) : 
    tensorflowNet = cv2.dnn.readNetFromTensorflow( pb_path )
    print( tensorflowNet)
    return tensorflowNet

def loadInference( pb_folder ) : 
    model = tf.saved_model.load( pb_folder, tags= [ tag_constants.SERVING ])
    infer = model.signatures['serving_default']

    print("load Inference")
    print( infer)
    return infer



def main( _argv ) : 
    # load tensorflow model
    pb_path = ".\saved_model"    

    #infer = loadInference( pb_path)

    capture  = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession( config = config)

    STRIDES, ANCHORS, NUM_CLASS, XYSCLE = utils.load_config(FLAGS)

    model = tf.saved_model.load( pb_path, tags= [ tag_constants.SERVING ])
    infer = model.signatures['serving_default']

    

    input_size = 416

    while cv2.waitKey( 1) < 0 : 
        ret, frame = capture.read()

        image = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB)
        scaled_image = cv2.resize( image, (input_size, input_size))
        
        scaled_image  = scaled_image / 255.
        image_data = scaled_image[np.newaxis, ...].astype(np.float32)

        """
        input_data = []

        for i in range(1) :
            input_data.append( scaled_image )
        input_data = np.asarray( input_data).astype(np.float32)
        """
        #boxes = []
        #pred_conf = []

        batch_data = tf.constant( image_data )        

        pred_bbox = infer( batch_data)

        for key, value in pred_bbox.items() : 
            boxes = value[ : , : ,  0 : 4]
            pred_conf = value[ :, : , 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes = tf.reshape( boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores = tf.reshape( pred_conf, (tf.shape( pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class = 50,
            max_total_size = 50, 
            iou_threshold = 0.45,#FLAGS.iou,
            score_threshold = 0.25#FLAGS.score
        )

        pred_bbox = [ boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy() ]
        output = utils.draw_bbox( frame, pred_bbox)
        #result = np.asarray(output)
        #result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


        cv2.imshow( "webcam", output)

    capture.release()
    cv2.destroyAllWindows()











if __name__ == '__main__' : 
    try : 
        app.run(main)
    except SystemExit:
        pass