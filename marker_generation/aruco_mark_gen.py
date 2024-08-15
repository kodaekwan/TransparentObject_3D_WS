import cv2
import numpy as np
import cv2.aruco as aruco


if __name__ == "__main__":
    
    board_type= aruco.DICT_6X6_250;
    MARKER_SIZE = 400;
    id_info = 8;
    
    arucoDict = aruco.getPredefinedDictionary(board_type);
    aruco_matker_img = aruco.generateImageMarker(arucoDict , id_info , MARKER_SIZE);
    
    cv2.imshow("aruco_matker_img",aruco_matker_img);
    cv2.waitKey(0);



