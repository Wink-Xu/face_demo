# Face Recognition Demo

A small demo to compare the the similarity of two image .

## Three process  

* face keypoints   
* face align   
* face feature

## Usage
get face keypoints, align face and get features
```
cd core
sh run.sh
```
Compare the similarity of two image
```
cd core
sh compare.sh
```


##

Run in cpu, if you want to run in GPU, set the use_gpu=1 in getKeypoint.py and getFeture.py.  (lazy to modify)