import cv2
import numpy as np
import d435cam
import time
import open3d as o3d
from sklearn.cluster import DBSCAN
from collections import Counter

def func_norm_vec(rvecs):



    data=np.array(rvecs).squeeze(-1);
    # Apply RANSAC to filter out outliers
    if(data.shape[0]<4):
        inlier_mask = np.zeros((data.shape[0]),dtype=np.bool_);
        inlier_mask[0] = True;
        return rvecs[0:1], inlier_mask

    db = DBSCAN(eps=0.25, min_samples=3).fit(data)
    labels = db.labels_

    #inlier_mask = labels != -1
    #inlier_vectors = data[inlier_mask]


    # 라벨의 빈도를 계산합니다.
    label_counts = Counter(labels)

    # 가장 빈도가 높은 라벨을 찾습니다.
    most_common_label = label_counts.most_common(1)[0][0]
    
    # 가장 빈도가 높고 라벨이 -1이 아닌 인덱스 마스크
    inlier_mask = (labels == most_common_label) & (labels != -1);


    # 가장 빈도가 높은 라벨에 해당하는 데이터를 필터링합니다.
    inlier_vectors = data[inlier_mask]

    return inlier_vectors,inlier_mask;

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
    
    # aruco detector 생성
    board_type = cv2.aruco.DICT_6X6_250;
    arucoDict  = cv2.aruco.getPredefinedDictionary(board_type);
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(arucoDict, parameters);
    
    #realsense 카메라 초기 노출시간 확보
    time.sleep(2)

    pre_cm_f = None;
    pre_avg_r = None;
    
    while(True):
        #realsense 카메라로부터 촬영 이미지 가져오기
        ret,img = cam.read()
        
        #파란색상 정의
        blue_BGR = (255, 0, 0)
        red_BGR = (0, 0, 255)
        green_BGR = (0, 255, 0)

        #마커사이즈 실제 사이즈와 공간 좌표(x,y,z) -> z가 0인 이유는 마커가 평면이기 때문이다.
        marker_size = 30.0
        marker_3d_edges = np.array([    [0,0,0],
                                        [0,marker_size,0],
                                        [marker_size,marker_size,0],
                                        [marker_size,0,0]], dtype = 'float32').reshape((4,1,3))
        center_object_point = np.mean(marker_3d_edges, axis=0).reshape((1, 3))
        
        if(ret):
            # 마커(marker) 검출
            corners, ids, rejectedCandidates = detector.detectMarkers(img.copy())


            R_list = [];
            T_list = [];

            if(ids is None):
                continue;
            
            pnp_results_dict = {};
            result_list = [];
            # 검출된 마커들의 꼭지점을 이미지에 그려 확인
            for corner,id in zip(corners,ids):
                corner = np.array(corner).reshape((4, 2))
                
                
            
                (topLeft, topRight, bottomRight, bottomLeft) = corner

                topRightPoint    = (int(topRight[0]),      int(topRight[1]))
                topLeftPoint     = (int(topLeft[0]),       int(topLeft[1]))
                bottomRightPoint = (int(bottomRight[0]),   int(bottomRight[1]))
                bottomLeftPoint  = (int(bottomLeft[0]),    int(bottomLeft[1]))

                #검출된 마커의 각 꼭지점 표시
                cv2.circle(img, topLeftPoint, 4, blue_BGR, -1)
                cv2.circle(img, topRightPoint, 4, blue_BGR, -1)
                cv2.circle(img, bottomRightPoint, 4, blue_BGR, -1)
                cv2.circle(img, bottomLeftPoint, 4, blue_BGR, -1)
                
                #ret, rvec, tvec = cv2.solvePnP(marker_3d_edges, corner, np.array(cmtx), np.array(dist));
                ret, rvec, tvec,_ = cv2.solvePnPRansac(marker_3d_edges, corner, np.array(cmtx), np.array(dist),reprojectionError=4.0);

                if(ret):
                    result_list.append([ret, rvec, tvec,id]);
                    # show 
                    x=round(tvec[0][0],2);
                    y=round(tvec[1][0],2);
                    z=round(tvec[2][0],2);
                    rx=round(np.rad2deg(rvec[0][0]),2);
                    ry=round(np.rad2deg(rvec[1][0]),2);
                    rz=round(np.rad2deg(rvec[2][0]),2);
                    # PnP 결과를 이미지에 그려 확인
                    text1 = f"{id}, {x},{y},{z}"
                    text2 = f"{rx},{ry},{rz}"
                    cv2.putText(img, text1, (int(topLeft[0]-10),   int(topLeft[1]+10)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
                    cv2.putText(img, text2, (int(topLeft[0]-10),   int(topLeft[1]+40)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
                    cv2.drawFrameAxes(img, np.array(cmtx), np.array(dist), rvec, tvec, marker_size)


            for ret, rvec, tvec, id in result_list:
                if(ret):
                    R_list.append(rvec);
                    

            # 평균적 회전방향 추정
            #avg_r = np.array(R_list).mean(0);

            inlier_data,inlier_idx = func_norm_vec(R_list);
            avg_r = np.array(inlier_data).mean(0);
            avg_r = avg_r.reshape(3);
            #print(inlier_idx , [result_[3][0] for result_ in result_list])
            #print(np.rad2deg(inlier_data))
            print(np.rad2deg(avg_r))
            


            for idx,(ret, rvec, tvec, id) in enumerate(result_list):
                
                if(inlier_idx[idx]==False):
                    continue;
                if(not ret):
                    continue;
                
                rvec = avg_r
                
                # 센터위치 추정
                center_image_point, _ = cv2.projectPoints(center_object_point, rvec, tvec, np.array(cmtx), np.array(dist))
                center_x, center_y = center_image_point.ravel()

                rmat, _ = cv2.Rodrigues(rvec)
                center_3d_position = rmat @ center_object_point.T + tvec

                imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0 ,0.0],]),rvec, center_3d_position.ravel(),np.array(cmtx), np.array(dist))
                x,y = np.abs(imgpts)[0][0].astype(np.uint32);
                #cv2.circle(img, (x,y), 3, red_BGR, -1)
                cv2.circle(img, (int(center_x),int(center_y)), 3, red_BGR, -1)

                #cv2.drawFrameAxes(img, np.array(cmtx), np.array(dist), rvec, center_3d_position.ravel(), 10)



                
                x,y,z = center_3d_position.ravel();
                #x,y,z = tvec.ravel()
                x_o,y_o,z_o = [0.0,0.0,0.0]
                h_offset = 140.0/2.0 + 30.0/2.0 # 140
                w_offset = 220.0/2.0 + 30.0/2.0 # 220

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
        


            if(len(T_list)==0):
                continue;
            # 중심점 계산
            #cm_ =  np.median(np.array(T_list),axis=0);
            cm_ =  np.array(T_list).mean(axis=0);
           
            #R, _ = cv2.Rodrigues()

            cm_f = np.float32([cm_]).transpose();
            
            #[0,0,0]
            gain = 0.0;
            if(pre_cm_f is None):
                pre_cm_f = cm_f.copy();
                pre_avg_r = avg_r.copy();
            else:
                pre_cm_f = (gain*pre_cm_f) + (1.0-gain)*cm_f;
                pre_avg_r = (gain*pre_avg_r) + (1.0-gain)*avg_r;



            #imgpts, jac = cv2.projectPoints(np.float32([[30.0/2,30.0/2 ,90.0],]),R_list[0],cm_f,np.array(cmtx), np.array(dist))
            imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0,90.0],]),pre_avg_r,pre_cm_f,np.array(cmtx), np.array(dist))
            x,y = np.abs(imgpts)[0][0].astype(np.uint32);
            cv2.circle(img, (x,y), 2, green_BGR, -1)

            # rx=round(np.rad2deg(pre_avg_r[0]),2);
            # ry=round(np.rad2deg(pre_avg_r[1]),2);
            # rz=round(np.rad2deg(pre_avg_r[2]),2);


     
            #print(R_list)



            #imgpts, jac = cv2.projectPoints(np.float32([[30.0/2,30.0/2 ,90.0-90],]),R_list[0],cm_f,np.array(cmtx), np.array(dist))
            imgpts, jac = cv2.projectPoints(np.float32([[0.0,0.0 ,0.0],]),pre_avg_r,pre_cm_f,np.array(cmtx), np.array(dist))
            x,y = np.abs(imgpts)[0][0].astype(np.uint32);
            cv2.circle(img, (x,y), 2, green_BGR, -1)
            
            text3 = "{:.2f}, {:.2f}, {:.2f}".format(round(pre_cm_f[0][0],2),round(pre_cm_f[1][0],2),round(pre_cm_f[2][0],2))
            pre_avg_r_deg=np.rad2deg(pre_avg_r);
            text2 = f"{round(pre_avg_r_deg[0],2)},{ round(pre_avg_r_deg[1],2) },{round(pre_avg_r_deg[2],2)}"
            cv2.putText(img, text3, (x-10,   y+20), cv2.FONT_HERSHEY_PLAIN, 1.0, green_BGR)
            cv2.putText(img, text2, (x-10,   y+40), cv2.FONT_HERSHEY_PLAIN, 1.0, green_BGR)


            #cv2.drawFrameAxes(img, np.array(cmtx), np.array(dist), avg_r, cm_f, 10)


            
            # imgpts, jac = cv2.projectPoints(pcd_array,avg_r,cm_f,np.array(cmtx), np.array(dist));
            # for imgpt in imgpts:
            #     x,y = np.abs(imgpt)[0].astype(np.uint32);
            #     cv2.circle(img, (x,y), 1, red_BGR, -1)
            
            cv2.imshow("img",img);
            key=cv2.waitKey(33);
            if(key&0xFF==ord('q')):
                break

    
    cam.release();
    cv2.destroyAllWindows();