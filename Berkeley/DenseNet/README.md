ST-DenseNet (method used in paper)

1.	Extend original dense-net architecture in paper 10
2.	Replace 2D kernels into 3D counterparts
3.	https://github.com/liuzhuang13/DenseNet
a.	Contains pre-trained models 
4.	Problem of large number of features?
a.	Set growth rate parameter to 24
b.	Bottleneck layer (1x1x1 conv layer before each dense block)
5.	The input to the model is a sequence of the cropped BBox images (resized to 100H× 100W) of the pedestrians tracked over the past 16 frames 
6.	Output is 2 softmax classification scores
7.	Also implemented memory efficient approach

 

Conv-net BASELINE approaches (for comparison purposes)

1.	Paper 8
a.	2D
b.	Only takes a singular bounding box as input
2.	Paper 9 (ALexNet convnet)
a.	2D convnet
b.	Can take multiple bounding boxes as input
c.	Extract features and feed to SVM classifier
3.	C3D
a.	Feed extracted features to LSTM model
b.	3D convolutions
c.	Contains pre-trained models (sports data)
d.	http://vlg.cs.dartmouth.edu/c3d/
e.	https://github.com/facebookarchive/C3D
f.	https://docs.google.com/document/d/1-QqZ3JHd76JfimY4QKqOojcEaf5g3JS0lNh-FHTxLag/edit


Order of performance:
1.	ST-densenet
2.	C3D
3.	Others

Other resources:

Code Matthew ran
https://github.com/gudongfeng/3d-DenseNet

Example of simple 3D convnet
https://www.machinecurve.com/index.php/2019/10/18/a-simple-conv3d-example-with-keras/#todays-dataset-3d-mnist


