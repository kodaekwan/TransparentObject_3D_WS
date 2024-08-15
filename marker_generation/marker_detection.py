import cv2
import numpy as np
import d435cam
import time
import open3d as o3d
from sklearn.cluster import DBSCAN
from collections import Counter
import uuid
import json


class EstimateCameraPoseUsingTargetBoard:
    #색상 정의
    blue_BGR = (255, 0, 0)
    green_BGR = (0, 255, 0)
    red_BGR = (0, 0, 255)
    

    def __init__(self,
                        camera_matrix = np.eye(3),
                        camera_distortion = np.zeros(5),

                        marker_size = 30.0,
                        markers_in_height = 140.0,
                        markers_in_width = 220.0,

                        board_type = cv2.aruco.DICT_6X6_250,
                        reprojectionError = 4.0,
                        show_info = False
                        
                        ) -> None:
        
        
        # aruco detector 생성
        arucoDict  = cv2.aruco.getPredefinedDictionary(board_type);
        parameters = cv2.aruco.DetectorParameters()
        self.detector   = cv2.aruco.ArucoDetector(arucoDict, parameters);
    
        #3차원 마커의 각 꼭지점 생성
        self.marker_3d_edges = np.array([   [0,0,0],
                                            [0,marker_size,0],
                                            [marker_size,marker_size,0],
                                            [marker_size,0,0]], dtype = 'float32').reshape((4,1,3));
        #3차원 마커의 각 중심점 생성
        self.marker_3d_center = np.mean(self.marker_3d_edges, axis=0).reshape((1, 3));



        self.reprojectionError = reprojectionError;
        self.marker_size = marker_size;
        self.markers_in_height = markers_in_height;
        self.markers_in_width = markers_in_width;

        self.show_info = show_info;
        self.temp_img = None;
    
        self.cmtx = camera_matrix;
        self.dist = camera_distortion;


    def cluster_vector(self,rvecs):

        data=np.array(rvecs).squeeze(-1);
        # Apply RANSAC to filter out outliers
        if(data.shape[0]<4):
            inlier_mask = np.zeros((data.shape[0]),dtype=np.bool_);
            inlier_mask[0] = True;
            return rvecs[0:1], inlier_mask

        db = DBSCAN(eps=0.25, min_samples=3).fit(data)
        labels = db.labels_

        # 라벨의 빈도를 계산
        label_counts = Counter(labels)

        # 가장 빈도가 높은 라벨을 찾기
        most_common_label = label_counts.most_common(1)[0][0]
        
        # 가장 빈도가 높고 라벨이 -1이 아닌 인덱스 마스크
        inlier_mask = (labels == most_common_label) & (labels != -1);

        # 마스크 라벨에 해당하는 데이터를 필터링
        inlier_vectors = data[inlier_mask]

        return inlier_vectors,inlier_mask;
        
    
    def estimation_center_pose(self,image):
        self.temp_img = image.copy();
        corners, ids, rejectedCandidates = self.detector.detectMarkers(self.temp_img);
        result_list = [];
        T_list = [];

        if(ids is None):
            return None,None

        for corner,id in zip(corners,ids):
            corner = np.array(corner).reshape((4, 2))
            ret, rvec, tvec,_ = cv2.solvePnPRansac( self.marker_3d_edges, 
                                                    corner, 
                                                    self.cmtx, 
                                                    self.dist,
                                                    reprojectionError=self.reprojectionError);
            
            # 검출이 정상이라면
            if(ret):
                result_list.append([rvec, tvec, id[0]]);
                
                # 정보 보여주기라면
                if(self.show_info):
                    # 이미지에 검출된 마커 꼭지점을 그리기
                    (topLeft, topRight, bottomRight, bottomLeft) = corner

                    topRightPoint    = (int(topRight[0]),      int(topRight[1]))
                    topLeftPoint     = (int(topLeft[0]),       int(topLeft[1]))
                    bottomRightPoint = (int(bottomRight[0]),   int(bottomRight[1]))
                    bottomLeftPoint  = (int(bottomLeft[0]),    int(bottomLeft[1]))

                    #검출된 마커의 각 꼭지점 표시
                    cv2.circle(self.temp_img, topLeftPoint,     4, self.blue_BGR, -1);
                    cv2.circle(self.temp_img, topRightPoint,    4, self.blue_BGR, -1);
                    cv2.circle(self.temp_img, bottomRightPoint, 4, self.blue_BGR, -1);
                    cv2.circle(self.temp_img, bottomLeftPoint,  4, self.blue_BGR, -1);
                    
                    # 위치와 방향 정보 분할 
                    x,y,z=round(tvec[0][0],2),round(tvec[1][0],2),round(tvec[2][0],2);
                    rx,ry,rz=round(np.rad2deg(rvec[0][0]),2),round(np.rad2deg(rvec[1][0]),2),round(np.rad2deg(rvec[2][0]),2);
                    
                    # 이미지에 정보 표시
                    cv2.putText(self.temp_img, f"{id}, {x},{y},{z}",    (int(topLeft[0]-10),   int(topLeft[1]+10)), cv2.FONT_HERSHEY_PLAIN, 1.0, self.red_BGR);
                    cv2.putText(self.temp_img, f"{rx},{ry},{rz}",       (int(topLeft[0]-10),   int(topLeft[1]+40)), cv2.FONT_HERSHEY_PLAIN, 1.0, self.red_BGR);

                    # 이미지에 좌표 표시
                    cv2.drawFrameAxes(self.temp_img, self.cmtx, self.dist, rvec, tvec, self.marker_size);

            
        # 여러 방향 벡터중 정상 밀도기반 클러스터링
        inlier_rvec, inlier_idx = self.cluster_vector([rvec for rvec, tvec, id in result_list]);

        # 정상 방향 벡터들의 평균값 계산
        avg_rvec = np.array(inlier_rvec).mean(0).reshape(3);

        # 중심 좌표 추정
        for idx, (rvec, tvec, id) in enumerate(result_list):
            
            if(inlier_idx[idx]==False):
                continue;
            

            # 방향과 좌표점으로부터 센터위치 추정
            rmat, _ = cv2.Rodrigues(rvec);
            center_3d_position = rmat @ self.marker_3d_center.T + tvec;


            # 마커들의 중심점 상대점 계산
            x,y,z = center_3d_position.ravel();
            h_offset = self.markers_in_height/2.0 + self.marker_size/2.0 # 140
            w_offset = self.markers_in_width/2.0 + self.marker_size/2.0 # 220

            if(id==1):
                x_o,y_o,z_o = rmat @ np.array([h_offset,w_offset,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])
            elif(id==2):
                x_o,y_o,z_o = rmat @ np.array([-h_offset,w_offset,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])
            elif(id==3):      
                x_o,y_o,z_o = rmat @ np.array([h_offset,-w_offset,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])
            elif(id==4):
                x_o,y_o,z_o = rmat @ np.array([-h_offset,-w_offset,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])
            elif(id==5):
                x_o,y_o,z_o = rmat @ np.array([0.0,w_offset,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])
            elif(id==6):
                x_o,y_o,z_o = rmat @ np.array([-h_offset,0.0,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])
            elif(id==7):
                x_o,y_o,z_o = rmat @ np.array([0.0,-w_offset,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])
            elif(id==8):
                x_o,y_o,z_o = rmat @ np.array([h_offset,0.0,0.0]).T
                T_list.append([x+x_o,y+y_o,z+z_o])


            
            if(self.show_info):          
                # 대상 중심점 계산
                center_image_point, _ = cv2.projectPoints(self.marker_3d_center, rvec, tvec, self.cmtx, self.dist)
                center_x, center_y = center_image_point.ravel()
                imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0 ,0.0],]),rvec, center_3d_position.ravel(),self.cmtx, self.dist);
                x,y = np.abs(imgpts)[0][0].astype(np.uint32);

                cv2.circle(self.temp_img, (int(center_x),int(center_y)), 3, self.red_BGR, -1);

        if(len(T_list)==0):
            return None,None;
        

        center_tvec =  np.array(T_list).mean(axis=0);
        center_tvec_t = np.float32([center_tvec]).transpose();

        if(self.show_info):

            imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0 ,0.0],]),avg_rvec,center_tvec_t,self.cmtx, self.dist)
            x,y = np.abs(imgpts)[0][0].astype(np.uint32);
            cv2.circle(self.temp_img, (x,y), 2, self.green_BGR, -1)

            avg_rvec_deg=np.rad2deg(avg_rvec);

            cv2.putText(self.temp_img, "{:.2f}, {:.2f}, {:.2f}".format(round(center_tvec_t[0][0],2),round(center_tvec_t[1][0],2),round(center_tvec_t[2][0],2)) , (x-10,   y+20), cv2.FONT_HERSHEY_PLAIN, 1.0, self.green_BGR)
            cv2.putText(self.temp_img,  f"{round(avg_rvec_deg[0],2)},{ round(avg_rvec_deg[1],2) },{round(avg_rvec_deg[2],2)}" , (x-10,   y+40), cv2.FONT_HERSHEY_PLAIN, 1.0, self.green_BGR)


            imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0,90.0],]),avg_rvec,center_tvec_t,self.cmtx, self.dist);
            x,y = np.abs(imgpts)[0][0].astype(np.uint32);
            cv2.circle(self.temp_img, (x,y), 2, self.green_BGR, -1)
        
        return center_tvec_t,avg_rvec
                    




    def show(self):
        if(not self.temp_img is None):
            cv2.imshow("result",self.temp_img);
            return cv2.waitKey(33);
        return None;

def convert_np_int32_to_int(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


if __name__ == "__main__":



    path="CAD_Data/transprant_cap.STL"

    # STL 파일을 읽기
    stl_mesh = o3d.io.read_triangle_mesh(path)

    # STL 파일을 성공적으로 읽었는지 확인
    if stl_mesh.is_empty():
        print("Failed to read the STL file.")
    else:
        # 기본 메시 정보 출력
        print(stl_mesh)

    stl_mesh.translate([-40.0,-40.0,0])
    R = stl_mesh.get_rotation_matrix_from_xyz((0, np.pi ,0))
    stl_mesh.rotate(R, center=(0, 0, 0))
    stl_mesh.translate([0.0,0.0,50])

    pcd = stl_mesh.sample_points_uniformly(number_of_points=500)

    pcd.translate([0,0,40.0]);

    pcd_array = np.asarray(pcd.points); 



    # 카메라 생성 및 카메라, 렌즈 파라메터 정의
    cam = d435cam.realsense_camera(720,1280,30);
    #cam = d435cam.realsense_camera(1080,1920,30,use_depth=False);

    intrinsics = cam.get_intrinsics();
    cmtx = [[intrinsics.fx,0.0,intrinsics.ppx],
            [0.0,intrinsics.fy,intrinsics.ppy],
            [0.0,0.0,1.0],];
    dist = intrinsics.coeffs;

    ecptb =  EstimateCameraPoseUsingTargetBoard(camera_matrix=np.array(cmtx),camera_distortion=np.array(dist),show_info=True);

    while(True):
        
            ret,img = cam.read()

            if(ret):

                center_tvec,center_rvec = ecptb.estimation_center_pose(img);
                if(not center_tvec is None):
                    imgpts, jac = cv2.projectPoints(pcd_array,center_rvec,center_tvec,np.array(cmtx), np.array(dist));
                    for imgpt in imgpts:
                        x,y = np.abs(imgpt)[0].astype(np.uint32);
                        cv2.circle(ecptb.temp_img, (x,y), 1, ecptb.red_BGR, -1)
                
                
                key = ecptb.show();
                if(key&0xFF==ord('q')):
                    break
                
                if(key&0xFF==ord('s')):
                    if(not center_tvec is None):
                        id_num = uuid.uuid1()
                        id_str = str(id_num);
                        offset_pixel = 10

                        cv2.imwrite(f"dataset_collect/test/images/{id_str}.png",img);
                        imgpts, jac = cv2.projectPoints(pcd_array,center_rvec,center_tvec,np.array(cmtx), np.array(dist));
                        imgpts = imgpts.astype(np.int32);
                        x_max = np.abs(imgpts)[:,0,0].max(0)+offset_pixel;
                        x_min = np.abs(imgpts)[:,0,0].min(0)-offset_pixel;
                        y_max = np.abs(imgpts)[:,0,1].max(0)+offset_pixel;
                        y_min = np.abs(imgpts)[:,0,1].min(0)-offset_pixel;

                        h,w = img.shape[:2]

                        x_max = np.clip(x_max,0,w-1);
                        x_min = np.clip(x_min,0,w-1);
                        y_max = np.clip(y_max,0,h-1);
                        y_min = np.clip(y_min,0,h-1);

                        # imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0, 40.0],]),center_rvec,center_tvec,np.array(cmtx), np.array(dist))
                        # bottom_x,bottom_y = np.abs(imgpts)[0][0].astype(np.int32);


                        imgpts, jac = cv2.projectPoints(np.float32([[40.0,0.0, 50.0],]),center_rvec,center_tvec,np.array(cmtx), np.array(dist))
                        top_x1,top_y1 = np.abs(imgpts)[0][0].astype(np.int32);

                        imgpts, jac = cv2.projectPoints(np.float32([[0.0, 40.0, 50.0],]),center_rvec,center_tvec,np.array(cmtx), np.array(dist))
                        top_x2,top_y2 = np.abs(imgpts)[0][0].astype(np.int32);

                        imgpts, jac = cv2.projectPoints(np.float32([[-40.0,0.0, 50.0],]),center_rvec,center_tvec,np.array(cmtx), np.array(dist))
                        top_x3,top_y3 = np.abs(imgpts)[0][0].astype(np.int32);

                        imgpts, jac = cv2.projectPoints(np.float32([[0.0,-40.0, 50.0],]),center_rvec,center_tvec,np.array(cmtx), np.array(dist))
                        top_x4,top_y4 = np.abs(imgpts)[0][0].astype(np.int32);


                        imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0 ,90.0],]),center_rvec,center_tvec,np.array(cmtx), np.array(dist))
                        bottom_x,bottom_y = np.abs(imgpts)[0][0].astype(np.int32);

                        

                        save_label = {"bboxes":[[x_min,y_min,x_max,y_max],],"keypoints" : [[[top_x1,top_y1,1],[top_x2,top_y2,1],[top_x3,top_y3,1],[top_x4,top_y4,1],[bottom_x,bottom_y,1] ],]}
                        
                        with open(f"dataset_collect/test/annotations/{id_str}.json", 'w') as f :
                            json.dump(save_label, f,default=convert_np_int32_to_int);
                    

                    
                    

                

    
    cam.release();
    cv2.destroyAllWindows();