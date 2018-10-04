# gaze_tracking
Gaze-Tracking based on head orientation and eye orientation  

Reference paper: "Monocular Free-head 3D Gaze Tracking with Deep Learning and Geometry Constraints" http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Monocular_Free-Head_3D_ICCV_2017_paper.pdf  

The angle of the gaze can be obtained by a combination of the angle of the head gesture and the angle of the eye turn. Here, we will crop out the face and eye parts according to the key points of the face, and then send them to the two subnets face_net and eye_net respectively. The two sub-networks respectively output the head and eye orientation angles, expressed by the longitude lo and the latitude la. Then combine these two angles to calculate the final gaze angle lo,la.  
  
## About Code
train_head&eye.py  
Train the head network to predict head orientation; Train the eye network to predict eye orientation.This code contains data augmentation.   
cal_gaze.py  
Combine head angles and eye angles to calculate the final gaze angle.  
direct_train_gaze.py  
Directly train the gaze network without head-net and eye-net. This may leads to overfitting.  
extract_feat.py  
extract landmark-points from face image using face-alignment (https://github.com/1adrianb/face-alignment)  
train_head_with_landmark.py  
Train the GBDT(light gbm) model to predict head orientation using landmark-points.  
  
## Results
On this data set, the error value of the angle of gaze is within 10 degrees.  

If you want to know more details, please read 'gaze_tracking_报告.pdf'(Chinese). If you want to read details in English ，please contact me.
