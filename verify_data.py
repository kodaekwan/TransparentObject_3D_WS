import cv2
import json
import os

data_path = "dataset_collect/train/";
#data_path = "/media/dkko/4a1a8a88-766b-44b9-8dcd-9bf89c92b299/keypoint_3d_pose_estimation_ws/keypoint_detection/data/glue_tubes_keypoints_dataset_134imgs/train/"

file_names = os.listdir(data_path+"annotations");

for file_name_ext in file_names:

    file_name, file_extension = os.path.splitext(file_name_ext);
    
    img = cv2.imread(data_path+"images/"+file_name+".png");
    with open(data_path+"annotations/"+file_name+".json",'r') as f:
        file = json.load(f);
        bboxes = file['bboxes'][0];
        keypoints = file['keypoints'][0];
        cv2.rectangle(img, bboxes[:2],bboxes[2:], (255,0,0));

        
        for keypoint in keypoints:
            cv2.circle(img, keypoint[:2], 3, (0,0,255), -1)
        
        cv2.imshow("img",img)
        cv2.waitKey(0)

    
