import numpy as np
import cv2

#we are using the class names from coco dataset, which has 80 different classes
classnames = []

with open('files/coco.names') as f:
    classnames = f.read().rstrip('\n').split('\n')

#import Deep SORT methods
import deep_sort.preprocessing
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import deep_sort.generate_detections as gdet

#creating neural network to detct objects
nnet = cv2.dnn.readNetFromDarknet('files/yolov3.cfg', 'files/yolov3.weights')
nnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
nnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    conf_thresh = 0.8
    nms_thresh = 0.3
    
    h_tar, w_tar, channels_tar = img.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for d in output:
            scores = d[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_thresh:
                w,h = int(d[2]*w_tar), int(d[3]*h_tar)
                x,y = (int(d[0]*w_tar) - w/2), (int(d[1]*h_tar) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                
    indices = cv2.dnn.NMSBoxes(bbox,confs,conf_thresh,nms_thresh)
    
    nms_bbox = []
    nms_confs = []
    nms_classIds = []
    
    for i in indices:
        i = i[0]
        nms_bbox.append(bbox[i])
        nms_confs.append(confs[i])
        nms_classIds.append(classIds[i])
  
    return nms_bbox, nms_classIds, nms_confs

#function to detect objects
def detect(nnet,img):
    w_h_tar = 320  # since useing yolo v3 320x320
    
    blob_img = cv2.dnn.blobFromImage(img,1/255,(w_h_tar,w_h_tar),[0,0,0],1,crop=False)
    nnet.setInput(blob_img)
    
    layerNames = nnet.getLayerNames()
    
    outputNames = [layerNames[i[0]-1] for i in nnet.getUnconnectedOutLayers()]
    
    outputs = nnet.forward(outputNames)
    
    return findObjects(outputs, img)

#get random colors
colors = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

#set model path
model_filename = 'files/market1501.pb'

#initialize encoder
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

#initializing the metric

#parameters for metric
max_cosine_distance = 0.5
nn_budget = None

metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

#initialising tracker
tracker = Tracker(metric)


#detecting and tracking objects
inp = int(input('Choose the format for detecting objects : \n 1.Video \n 2.Webcam \n'))

if inp == 1: #for video
    cap = cv2.VideoCapture('data/video00.mp4')
elif inp == 2: #for Webcam
    cap = cv2.VideoCapture(0)

    
while True:
    success, img = cap.read()
    
    #detecting objects and returning bounding boxes, class_ids, and confidence values
    boxes, class_ids, scores = detect(nnet,img)
    
    #getting the classnames from the class_ids
    names=[]
    for i in class_ids:
      names.append(classnames[i])
    
    #getting the features of the detected objects
    features = np.array(encoder(img,boxes))
    
    #detection list containing information of the bounding box, confidence value, classname, features for each detected object
    detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]
    
    #calling our DeepSORT tracker to update the tracker with the new detections and predict whether they were present before
    tracker.predict()
    tracker.update(detections)
    
    # i is required to generate different colors for the bounding box of different objects
    i = int(0)

    indexIDs = []

    for track in tracker.tracks:
      if not track.is_confirmed() or track.time_since_update > 1:
          continue

      #tracking ID
      indexIDs.append(int(track.track_id))

      #bounding box
      bbox = track.to_tlbr()

      #generating a color
      color = [int(c) for c in colors[indexIDs[i] % len(colors)]]

      if len(names) > 0:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
        cv2.putText(img,names[0].title()+" "+str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)

      i += 1
    
    cv2.imshow('Vid',img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
