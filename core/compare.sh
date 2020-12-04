## Get Keypoints, Align Face and Extract Feature .


IMG_PATH1=../sample/0bb7abab-84e2-40de-95d8-d68750640cd9-20190520120946-face-0.jpg  # Wang
IMG_PATH1=../sample/058d5d96-5ab3-4619-b162-45acc4575c06-20190516124507-face-0.jpg  # Luo
IMG_PATH2=../sample/0d7a9e79-4929-45e1-9bcf-95e7682f83f6-20190514231127-face-0.jpg  # Wang
IMG_PATH2=../sample/013a7277-d074-442e-9af2-c06628878636-20190527085159-face-0.jpg  # Luo

python faceRecognition.py --img_first $IMG_PATH1 --img_second $IMG_PATH2