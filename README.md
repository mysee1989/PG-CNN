# PG-CNN
Code for our ICPR 2018 paper: ["Patch-Gated CNN for Occlusion-aware Facial Expression Recognition"](http://vipl.ict.ac.cn/uploadfile/upload/2018092516364248.pdf)

The following figure shows how we select the ciritcal points to crop patches from a facial image:
![alt text](https://github.com/mysee1989/PG-CNN/blob/master/img/point.png)   
(a) denotes totally 30 landmarks within original 68 facial landmarks. These points are involved in point selection   
(b) shows 16 points that we pick to cover the facial regions on or around eyes, eyebrows, nose, mouth   
(c) illustrates four points that are recomputed to better cover eyes and eyebrows   
(d) displays four points that are re-computed to cover facial cheeks   
(e) shows the selected 24 facial landmarks, around which the patches in (f) are cropped   


**Training yourself**       
<br />We designed caffe layer named by ***multi_roi_pooling_layer***. Currently the layer is provided with a GPU version.   
Building the ***multi_roi_pooling_layer*** with the related ***caffe.proto*** in proto folder, you can start training a model immediately.

**Precautions**       
<br />The training and testing image list should be arranged as    
 <br />&emsp;&emsp;image_path  expression_label  point1_h  point1_w  pint2_h point2_w  ...   point_24_h  point24_w   
 A example train.list has been provided in train_list_example folder
 <br /> We provided a python script to convert 68 facial landmarks to desired 24 points, the file locate in ***convert_point*** folder
