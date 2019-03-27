# PG-CNN and its TIP version
By [Yong Li](https://www.linkedin.com/in/%E5%8B%87-%E6%9D%8E-350025105/), Jiabei Zeng, Shiguang Shan, Xilin Chen
<br /><br /> News! 2019/02/25.  We add the model config of paper: ["Occlusion aware facial expression recognition using CNN with attention mechanism"](https://ieeexplore.ieee.org/abstract/document/8576656). All the details can be found in ***prototxt/gACNN_train.prototxt***.

<br />Code for our ICPR 2018 paper: ["Patch-Gated CNN for Occlusion-aware Facial Expression Recognition"](http://vipl.ict.ac.cn/uploadfile/upload/2018092516364248.pdf). We designed a Patch-Gated CNN that can percept and ignore the occlusions for facial expression recognition. All the details can be found in ***prototxt/pACNN_train.prototxt***. 
<br />Note that the code is based on [caffe](https://github.com/BVLC/caffe), a famous deep learning framework.

The order of the 68 facial landmarks can be found at [Link](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/).<br />
The following figure shows how we select the ciritcal points to crop the patches from a facial image:
![alt text](https://github.com/mysee1989/PG-CNN/blob/master/img/point.png)   
(a) denotes totally 30 landmarks within original 68 facial landmarks. These points are involved in point selection   
(b) shows 16 points that we pick to cover the facial regions on or around eyes, eyebrows, nose, mouth   
(c) illustrates four points that are recomputed to better cover eyes and eyebrows   
(d) displays four points that are re-computed to cover facial cheeks   
(e) shows the selected 24 facial landmarks, around which the patches in (f) are cropped   


**Training yourself:**       
&emsp;&emsp;We designed a caffe layer named as ***multi_roi_pooling_layer***. Currently the layer is provided with a GPU version.   
<br />&emsp;&emsp;Building the ***multi_roi_pooling_layer*** with the related ***caffe.proto*** in proto folder, you can start training a model <br />&emsp;&emsp; immediately.

<br />**Precautions:**       
&emsp;&emsp;The training and testing image list should be arranged as:    
 &emsp;&emsp;&emsp;&emsp;image_path &emsp; expression_label &emsp; point1_h &emsp; point1_w &emsp; pint2_h &emsp; point2_w &emsp; ... &emsp;  point_24_h &emsp; point24_w   
 <br />&emsp;&emsp;An example of ***train.list*** has been provided in train_list_example folder.
 <br />&emsp;&emsp;We provided a python script to convert 68 facial landmarks to desired 24 points, the file locate in ***convert_point*** folder.
 <br /><br /><br />**Dateset resource:**
 <br />&emsp;&emsp;We collected and labelled a facial expression dataset (***FED-RO***) in the presence of real occlusions: . [download link](https://1drv.ms/u/s!AjMhxexGSrsZgQEy31L0HDGnXJjZ)
<br />&emsp;&emsp;Alternative [download link](https://pan.baidu.com/s/1kLKkClTnrbfY9hJr6shkHQ) based on Baidu Yun. Extraction code: o5di
