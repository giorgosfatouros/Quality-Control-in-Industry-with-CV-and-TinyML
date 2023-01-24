# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video casting_test.mp4 --tracker mil --model cnn_model_s

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import tflite_runtime.interpreter as tf


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="mil",
                help="OpenCV object tracker type")
ap.add_argument("-m", "--model", type=str, default="cnn_model_s.tflite",
                help="Name of tf lite model. It should be in the saved_models folder")
args = vars(ap.parse_args())

# Load the TFLite model and allocate tensors.
interpreter = tf.Interpreter(model_path=f"saved_models/{args['model']}")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

class_labels = ['Deficient', 'Normal']
# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        # "csrt": cv2.TrackerCSRT_create,
        # "kcf": cv2.TrackerKCF_create,
        # #		"boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        # "tld": cv2.TrackerTLD_create,
        # "medianflow": cv2.TrackerMedianFlow_create,
        # "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going to track
initBB = None

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] starting test video...")
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    time.sleep(0.5)
    # check to see if we have reached the end of the stream
    if frame is None:
        print('End of video')
        break

    # resize the frame (so we can process it faster) and grab the frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            track = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

        # update the FPS counter
        fps.update()
        fps.stop()
        # # initialize the set of information we'll be displaying on the frame
        # info = [
        #     ("Tracker", args["tracker"]),
        #     ("Success", "Yes" if success else "No"),
        #     ("FPS", "{:.2f}".format(fps.fps())),
        # ]
        #
        # # loop over the info tuples and draw them on our frame
        # for (i, (k, v)) in enumerate(info):
        #     text = "{}: {}".format(k, v)
        #     cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Resize to respect the input_shape
        tracked = cv2.resize(track, (300, 300))
        # Convert img to RGB
        # img = cv2.cvtColor(tracked, cv2.COLOR_GRAY2RGB)
        # img = img / 255
        # img = tracked.reshape(300, 300, 3)
        img = tracked.astype('float32')
        # input_details[0]['index'] = the index which accepts the input
        # 'output' is dictionary with all outputs from the inference.
        interpreter.set_tensor(input_details[0]['index'], [img])
        # run the inference
        interpreter.invoke()
        # output_details[0]['index'] = the index which provides the input
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_label = class_labels[output_data.argmax()]
        pred_scores = output_data.max()
        if pred_label == 'Deficient':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        info = [
            ("Label", pred_label),
            ("Probability", "{:.2f}".format(pred_scores)),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
