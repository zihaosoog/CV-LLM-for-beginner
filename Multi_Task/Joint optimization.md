### Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
**homoscedastic uncertainty**  
多任务包括回归和分类任务  
回归任务：逐像素深度估计（热力图回归）每个像素回归距离  采用**高斯似然估计**  
分类任务：语义分割以及实例分割 （每个像素进行多分类） 采用**对数似然估计**  


- [ ] GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
- [ ] Dynamic Task Prioritization for Multitask Learning
- [ ] End-to-End Multi-Task Learning with Attention
- [ ] MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning



### OmniDet: Surround View Cameras based Multi-task Visual Perception Network for Autonomous Driving
**VarNorm**

![Screenshot from 2022-11-24 16-03-27](https://user-images.githubusercontent.com/67272893/203726745-3ad0c0e5-fdf6-412a-9d4e-efc48329c33c.png)

> The loss weight of task i at epoch t, where $\bar{L_i}$ is the average of task loss i over the last n epochs.
```
把 第i个task 的 相邻n个epoch 的 loss_i_t 作为一个集合，集合中每一个数值都是在均值附近左右波动
通过 方差归一化 后的损失的倒数来当作权重，平衡不同task
```
