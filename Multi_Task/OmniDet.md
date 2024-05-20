# OmniDet: Surround View Cameras based Multi-task Visual Perception Network for Autonomous Driving
自动驾驶中鱼眼相机与常规相机相辅相成，共同完成信息融合，其中鱼眼相机具有径向失真，视野减小和周围特征失真。

## 平衡不同任务之间损失  
[Joint optimization](https://github.com/zh-song/Object-Detection-Papers/blob/Docments/Multi%20task/Joint%20optimization.md)

## 网络结构
1. 多个单目相机深度估计之间的泛化问题  
**单视图（*single view*）深度估计**存在一个**问题**，用camera_A收集到的图片来训练得到的深度估计网络model_A，不能在其他摄像机拍摄的图片上进行测试，因为相机参数等存在不同. 为了能在不同相机中使用训练好的网络，许多工作致力于此，例如**CAM-Convs**.  
> This paper **converts** all camera geometry properties **into** a tensor called ***camera geometry tensor Ct*** that is then passed to the CNN model to tackle this problem. It is included in each self-attention stage and also applied to every skip-connection.
2. 自监督深度估计中运动目标的无限深度问题  
采用分割mask滤除运动目标，通过静态物体获取深度信息
3. semanic segmentation guide distance prediction
4. 共享encoder和协同decoder


![Screenshot from 2022-11-25 15-24-19](https://user-images.githubusercontent.com/67272893/203924239-c36d2ef2-e29b-4170-829e-4798ae9c3287.png)






















> Reference  

[1]. [FisheyeMultiNet: Real-time Multi-task Learning Architecture for Surround-view Automated Parking System](https://github.com/zh-song/Object-Detection-Papers/blob/Docments/Multi%20task/FisheyeMultiNet.md#fisheyemultinet-real-time-multi-task-learning-architecture-for-surround-view-automated-parking-system)
