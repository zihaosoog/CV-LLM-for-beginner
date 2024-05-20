## FisheyeMultiNet: Real-time Multi-task Learning Architecture for Surround-view Automated Parking System.
> 任务目标
```
自动泊车
```
> 方案
```
1. 从四个摄像头中检测到的任何物体都被记录在图像坐标中，映射到世界坐标以创建共同的表示，并被馈送到虚拟地图中以计划自动停车的汽车操纵。
2. 道路标记和路缘石的处理方式相同，也被发送到地图上，为我们周围的世界建立一个可行的模型。
3. 通过假设一个平坦的地面，并使用车辆和摄像机校准将脚点（物体与地面的交点）映射到世界位置，可以在行人和车辆等物体周围建立边界框。
4. 深度估计可以处理脚点被遮挡或道路不平坦的情况。
```
Multi-task：
> The main advantage of a multitask network is its high computational efficiency, which is most suitable for a low cost embedded device.
> Reusing the encoder also provides regularization across different tasks.

![image](https://user-images.githubusercontent.com/67272893/203465667-4ea8da0a-a60c-48bc-aed6-d74161c7323a.png)

1. object detection（以几何线索辅助实现，包括运动分割，深度估计）
```
decoder: The object detection decoder is built using a grid level softmax layer
```
2. semantic segmentation 语义分割
```
decoder: FCN8 decoder with skip connections
loss: 分类交叉熵（categorical cross entropy）
```
3. soiling detection 镜头油污检测（鱼眼相机安装位置较低，易被路面泥水所污染）
```
task: the mixed multilabel-categorical classification problem based grid of image
decoder: the soiling decoder is built using a grid level softsign layer.
loss: 分类交叉熵（categorical cross entropy）
```
> *Tips*： 总损失为各task损失的加权之和(各任务损失值量级不同,难以同时收敛)，单个任务损失为平均值

多任务之间损失如何平衡加权：GradNorm为多任务学习中常用的loss融合方法
> We weigh the different tasks based on gradients observed after every epoch in a similar fashion to GradNorm.
