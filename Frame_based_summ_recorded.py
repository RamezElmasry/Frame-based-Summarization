import cv2
import numpy as np
import copy
import datetime as dt
import time
from cv2 import VideoWriter, VideoWriter_fourcc
import shutil
import os
from tracker.centroidtracker import CentroidTracker

# .....................Frame-Based Variables...................
# an array to show the detections with the higher confidence only in each frame
valid_detections = []
# array of tracked objects in current frame
current_frame_objects = []
# array of next frame objects
next_frames_objects = []
# background to put summary on
background_frame = None
# original video frame indices
original_frame_idx_arr = []

def doOverlap(box1, box2):
    [l1x, l1y, r1x, r1y, conf, label] = box1
    [l2x, l2y, r2x, r2y] = box2

    # To check if either rectangle is actually a line
    # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}

    if (l1x == r1x or l1y == r1y or l2x == r2x or l2y == r2y):
        # the line cannot have positive overlap
        return False

    # If one rectangle is on left side of other
    if (l1x >= r2x or l2x >= r1x):
        return False

    # If one rectangle is above other
    if (r1y <= l2y or r2y <= l1y):
        return False

    return True


def add_valid_detection(box, conf, label):
    temp = copy.deepcopy(valid_detections)
    valid_detections.clear()

    box.extend([conf, label])
    max_conf_detection = copy.deepcopy(box)

    for detection in temp:
        [startX, startY, endX, endY, conf0, label0] = detection
        check_intersection = doOverlap(box, [startX, startY, endX, endY])
        if check_intersection:
            max_conf_detection = detection if conf0 > max_conf_detection[4] else max_conf_detection
        else:
            valid_detections.append(detection)

    valid_detections.append(max_conf_detection)

def get_timestamp(frame, fps):
    total_seconds = int((frame + 1) // fps)
    return str(dt.timedelta(seconds=total_seconds))


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 10
    return 1


def sumVid():
    start = time.time()

    toughSummarization = False  # Summarization Type
    frame_basedSumm = True

    checkFrame = True  # Boolean variable used to later to drop every other frame

    # open the input video that will be summarized
    vs = cv2.VideoCapture(input_path)

    # Initialize the Mixture of Gaussian(MoG) Background subtractor with shadow detection and elimination by
    # setting the shadow value to zero.
    backGroundSub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True)
    backGroundSub.setShadowValue(0)
    backGroundSub.setNMixtures(5)

    # get the video details (frame width and height, frames per second (FBS) and the total number of frames)
    # the method of getting theses details is different in the version before and after openCV ver 3.0
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    if int(major_ver) < 3:
        width1 = vs.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = vs.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        FPS = vs.get(cv2.cv.CV_CAP_PROP_FPS)
        frames = vs.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    else:
        width1 = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
        FPS = vs.get(cv2.CAP_PROP_FPS)
        frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Total Frames:" + str(frames))
    SFrames = 0
    TFrames = 0
    minArea = width1 * height * 0.003
    print("Min Area = " + str(minArea))

    # initialize the output video that will contain the summarized video
    fourcc = VideoWriter_fourcc(*'mp4v')
    if toughSummarization and not frame_basedSumm:
        result = VideoWriter(f'{output_path}/output_tough_summarized.mp4', fourcc, FPS,
                             (int(width1 / 2), int(height / 2)),
                             0)
    if not toughSummarization and not frame_basedSumm:
        result1 = VideoWriter(f'{output_path}/output_normal_summarized.mp4', fourcc, FPS,
                              (int(width1), int(height)))

    if frame_basedSumm:
        result1 = cv2.VideoWriter("./temp/normal_summary.mp4",
                                  fourcc, FPS, (int(width1), int(height)))

    frame_idx = 0
    # loop over all frames in the input video.
    while True:
        # read a frame from the input video
        _, coloredFrame = vs.read()

        # if there is no more frames in the video then close the input and output video
        if coloredFrame is None:
            break
        # change the frame to grayscale since the results when using a colored frame are much worse than using a
        # grayscale one and that's because of the illumination changes becoming more pronounced and clear which
        # confuses the movement detection process
        frame = cv2.cvtColor(coloredFrame, cv2.COLOR_BGR2GRAY)

        tmp = frame.copy()

        # Apply a gaussian blur to the frame to smooth the edges
        tmp = cv2.GaussianBlur(tmp, (5, 5), 0)

        # preform a background subtraction and update the background model
        fgMask = backGroundSub.apply(tmp)

        # threshold the resultant frame to remove the values less than 150
        th_delta = cv2.threshold(fgMask, 150, 255, cv2.THRESH_BINARY)[1]

        # apply a close operation to fill the empty spots in the detected movement shapes
        th_delta = cv2.morphologyEx(th_delta, cv2.MORPH_CLOSE, None)

        # dilate the image to increase the size of the detected shapes
        th_delta = cv2.dilate(th_delta, None, iterations=2)

        # erode the image again to avoid the size of the detected objects from becoming too big
        # also a dilation followed by an erosion is similar to a closure which will further enhance the results
        th_delta = cv2.erode(th_delta, None, iterations=2)

        # detect the contours of the shapes detected due to movement in the frame
        Contours, _ = cv2.findContours(th_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # .setValue()
        # if the frame contains contours(Movement) then save it to the output video depending on the
        # summarization type. Tough summarization is the same as the normal summarization but with a reduced
        # frame dimensions and dropped frames

        rects = []

        for contour in Contours:
            if cv2.contourArea(contour) > minArea:
                (x, y, w, h) = cv2.boundingRect(contour)
                rects.append((x, y, x + w, y + h))

        if len(rects) != 0:
            if toughSummarization:
                if checkFrame:
                    frame1 = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    result.write(frame1)
                    # result1.write(coloredFrame)
                    TFrames += 1
                    SFrames += 1
                    checkFrame = False
                else:
                    checkFrame = True
                    # result1.write(coloredFrame)
                    SFrames += 1
            else:
                original_frame_idx_arr.append(frame_idx)
                result1.write(coloredFrame)

        frame_idx += 1

    vs.release()

    if toughSummarization:
        result.release()
    else:
        result1.release()

    endWithoutCompression = time.time()

    end = time.time()
    print("Runtime of the program is " + str(endWithoutCompression - start))


def create_background(vid_name):
    cap = cv2.VideoCapture(vid_name)
    # Randomly select 30 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)
    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)

        ret, frame = cap.read()

        frames.append(frame)

    # Calculate the median along the time axis
    global background_frame
    background_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cap.release()



def check_object_exists( id):
    for idx, object in enumerate(current_frame_objects):
        if object["id"] == id:
            return idx
    return -1



def add_new_detection( id, bbox, frame):
    idx = check_object_exists(id)
    if idx == -1:
        new_object = {
            "id": id,
            "leftX": np.array([bbox[0]]),
            "leftY": np.array([bbox[1]]),
            "rightX": np.array([bbox[2]]),
            "rightY": np.array([bbox[3]]),
            "frame": np.array([frame])
        }
        current_frame_objects.append(new_object)
    else:
        current_frame_objects[idx]["leftX"] = np.append(current_frame_objects[idx]["leftX"], bbox[0])
        current_frame_objects[idx]["leftY"] = np.append(current_frame_objects[idx]["leftY"], bbox[1])
        current_frame_objects[idx]["rightX"] = np.append(current_frame_objects[idx]["rightX"], bbox[2])
        current_frame_objects[idx]["rightY"] = np.append(current_frame_objects[idx]["rightY"], bbox[3])
        current_frame_objects[idx]["frame"] = np.append(current_frame_objects[idx]["frame"], frame)
        # sorting the leftX array after the new detection and reflecting on the other arrays
        inds = np.argsort(current_frame_objects[idx]["leftX"])
        current_frame_objects[idx]["leftX"] = current_frame_objects[idx]["leftX"][inds]
        current_frame_objects[idx]["leftY"] = current_frame_objects[idx]["leftY"][inds]
        current_frame_objects[idx]["rightX"] = current_frame_objects[idx]["rightX"][inds]
        current_frame_objects[idx]["rightY"] = current_frame_objects[idx]["rightY"][inds]
        current_frame_objects[idx]["frame"] = current_frame_objects[idx]["frame"][inds]


def detect(source):
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("./mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                   "./mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

    # initialize the tracker
    ct = CentroidTracker()

    vs = cv2.VideoCapture(source)
    frame_idx = 0

    while True:
        (grabbed, frame) = vs.read()
        if frame is None:
            break

        # frame = cv2.resize(frame, (1280,720))
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                if label != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                add_valid_detection([startX, startY, endX, endY], confidence, label)

        # initialize the list of bounding box rectangles to be tracked
        rects = []

        for detection in valid_detections:
            [startX, startY, endX, endY, conf0, label0] = detection
            rects.append([startX, startY, endX, endY])

        valid_detections.clear()

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        tracked_objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, rect) in tracked_objects.items():
            text = "ID {}".format(objectID)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.putText(frame, text, (rect[0], rect[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            add_new_detection(objectID, rect, frame_idx)

        frame_idx += 1
    vs.release()



def create_temp():
    temp_path = os.path.join(os.getcwd(), "temp")
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)



def delete_temp():
    temp_path = os.path.join(os.getcwd(), "temp")
    shutil.rmtree(temp_path)



def remove_from_current_frame_objects( id):
    idx = -1
    for i, object in enumerate(current_frame_objects):
        if object["id"] == id:
            idx = i

    return current_frame_objects.pop(idx)



# get index of first leftX that is greater than or equal to current section
def first_leftX_idx(  object_idx, low, high, section_X):
    if (high >= low):
        mid = low + (high - low) // 2
        if ((mid == 0 or section_X >  current_frame_objects[object_idx]["leftX"][mid - 1]) and
                 current_frame_objects[object_idx]["leftX"][mid] >= section_X):
            return mid
        elif (section_X > current_frame_objects[object_idx]["leftX"][mid]):
            return first_leftX_idx(object_idx, (mid + 1), high, section_X)
        else:
                return first_leftX_idx(object_idx, low, (mid - 1), section_X)
        return -1


def get_timestamp(frame, fps):
    total_seconds = int((frame +1) // fps)
    return str(dt.timedelta(seconds = total_seconds))


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1


def summarize(source):
    vid_cap = cv2.VideoCapture(source)
    n = 1
    object_no_timeStamp = ""
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    object_number = 1
    print(len(current_frame_objects))

    #loop until there are no objects left to be summarized
    while (len(current_frame_objects) != 0) or (len(next_frames_objects) != 0):
        #print("\n after loop inside summarize")
        #print("\n", next_frames_objects)
        current_frame_objects.extend(next_frames_objects)
        next_frames_objects.clear()
        current_section_X = 0
        background_path = output_path + '/' + str(n) + ".jpg"
        #make a copy of background frame
        background = copy.deepcopy(background_frame)

        #loop until there are no objects left to be put in the current frame
        #each iteration in this loop should assign an object to a different section
        while len(current_frame_objects) != 0:
            most_left_object_idx = -1
            smallest_leftX = 10000
            smallest_leftX_idx = -1
            next_section_X = -1
            next_Frame_ids = []

            #loop over available objects to assign the most left object to the current section
            for idx in range(len(current_frame_objects)):
                #get indices where leftX of an object is greater than or equal to current_section_X
                #indices_arr = np.where(object["leftX"] >= current_section_X)[0]

                index = first_leftX_idx(idx, 0, len(current_frame_objects[idx]["leftX"]) - 1, current_section_X)
                #if it happens that there is no value in the indicies array that can be put in
                #current frame we will add this object to the next_frame_inds to be added in
                # the next frame since there is a collision.
                if index != -1:
                    first_leftX = current_frame_objects[idx]["leftX"][index]

                    if first_leftX < smallest_leftX:
                        most_left_object_idx = idx
                        smallest_leftX = first_leftX
                        smallest_leftX_idx = index
                        next_section_X = current_frame_objects[idx]["rightX"][index]

                #if an object does not have a leftX that is greater than or equal to current_section_X
                #then there is a collision and the object needs to be assigned to another frame
                else:
                    next_Frame_ids.append(current_frame_objects[idx]["id"])

            if most_left_object_idx != -1:
                #crop object and put it in background frame
                leftX = current_frame_objects[most_left_object_idx]["leftX"][smallest_leftX_idx]
                leftY = current_frame_objects[most_left_object_idx]["leftY"][smallest_leftX_idx]
                rightX = current_frame_objects[most_left_object_idx]["rightX"][smallest_leftX_idx]
                rightY = current_frame_objects[most_left_object_idx]["rightY"][smallest_leftX_idx]
                frame_idx = current_frame_objects[most_left_object_idx]["frame"][smallest_leftX_idx]

                vid_cap.set(1, int(frame_idx))
                _ , frame = vid_cap.read()
                crop = copy.deepcopy(frame[int(leftY):int(rightY), int(leftX):int(rightX)])
                background[int(leftY):int(rightY), int(leftX):int(rightX)] = crop

                timestamp = get_timestamp(original_frame_idx_arr[frame_idx], fps)
                object_no_timeStamp += str(object_number) + " : " + timestamp + "\n"
                scale = get_optimal_font_scale(str(object_number), int(rightX) - int(leftX))
                cv2.putText(background, str(object_number), (int(leftX), int(leftY)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), thickness = 1)
                object_number += 1

                current_section_X = next_section_X
                remove_from_current_frame_objects(current_frame_objects[most_left_object_idx]["id"])

            #Since now we have put all the possible objects in their respective frames, we transfer
            # all the remaining objects to the next_frames_objects array, as we already saved these
            # objects id in the next_frames_ids array
            for i in next_Frame_ids:
                next_frames_objects.append(remove_from_current_frame_objects(i))

        #save current background frame to output folder
        cv2.imwrite(background_path, background)
        n+=1

    timestamps_path = output_path + "/timestamps.txt"
    with open(timestamps_path, 'w+') as f:
        f.write(object_no_timeStamp)

    vid_cap.release()


def frame_based_SUMM(input_path):
    create_temp()

    print("Normal Summarization starting")
    sumVid()

    print("creating background")
    create_background(input_path)

    source = "./temp/normal_summary.mp4"

    print("Detecting Objects")
    detect(source)

    print("Frame-based Summarization")
    summarize(source)

    delete_temp()

if __name__ == '__main__':
    # Function call Takes input video path
    input_path = 'hall_video.avi'
    output_path = './output_test'
    frame_based_SUMM(input_path)