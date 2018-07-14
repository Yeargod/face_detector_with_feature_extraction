###Dependency###
The operation system and software versions I used are listed as follows for your reference.

Ubuntu: 14.04
Cuda: 8.0
CuDNN: 5.1.10
OpenCV: 2.4.13

###Prediction:###
If you only want to predict your own face images with existing trained model, follow the steps below:

1. Go to the folder "YOLO_detector" and run 'make'

2. Download trained model from https://drive.google.com/open?id=1526kBv0tT09SyspwR-39V5wCS_AgZu5Z or https://pan.baidu.com/s/1xelCkgDZ85qNB-XNS1P6Eg

3. Run 'demo_run_face_detector.py'. It can run at 80fps using one 1080Ti GPU.

###Training###

1. Go to the folder "YOLO_detector" and run 'make'

2. Prepare your training images and ground-truth bounding box labels. The path of training images is saved in cfg/face.data. The default paths of bounding box labels can be obtained by replacing  'YOLOv2_training_images' in image paths with 'YOLOv2_training_labels', as specified in src/data.c.  One label file example is label_sample.jpg.txt, in which each row contains class_id and (centerx,centery,w,h).   Note that (centerx,centery,w,h) are normalized to (0,1). Since we only have one class, so all class_ids are 0. 

3. Download the initial model 'darknet19_448.conv.23' pretrained on ImageNet from https://drive.google.com/open?id=1526kBv0tT09SyspwR-39V5wCS_AgZu5Z or https://pan.baidu.com/s/1xelCkgDZ85qNB-XNS1P6Eg

4. After training images and labels are ready, run './train.sh', and the learnt models will be saved in the folder "backup".




