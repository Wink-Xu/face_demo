## Get Keypoints, Align Face and Extract Feature .


#IMG_PATH=../sample/0bb7abab-84e2-40de-95d8-d68750640cd9-20190520120946-face-0.jpg  # Wang
IMG_PATH=../sample/013a7277-d074-442e-9af2-c06628878636-20190527085159-face-0.jpg  # Luo
##### First, you should get face keypoints,  we will put the result '.json' and '_kp.jpg' in ../results directory.
python getKeypoint.py --img_path $IMG_PATH

##### Second, you should align the face through the face keypoints, we will put the result '_align.jpg' in ../results directory.
#python alignFace.py --img_path $IMG_PATH 

##### Last, you can get the face feature
#python getFeature.py --img_path $IMG_PATH 
