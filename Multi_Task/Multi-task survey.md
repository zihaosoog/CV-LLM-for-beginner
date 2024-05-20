# Multi-Task Learning for Dense Prediction Tasks: A Survey

> Multi-Task Learning (MTL) aims to improve such generalization by leveraging domain-specific information contained in the training signals of related tasks. 

**多任务实际应用价值**：  

```
自动驾驶 同时要求 车道线分割，场景中目标检测，估计距离与轨迹等
```

**多任务学习优点**： 
```
1. Due to their inherent layer sharing, the resulting memory footprint is substantially reduced (内存减少)
2. As they explicitly avoid to repeatedly calculate the features in the shared layers, once for every task, they show increased inference speeds (推理速度加快)
3. They have the potential for improved performance if the associated tasks share complementary information, or act as a regularizer for one another (共享信息,相互充当正则项，提高精度)
```
## **Shared Encoder**   

经常搭配 **independent task-specific head **

1. hard share  

```
特点：a shared encoder that branches out into task-specific decoding heads 
问题：In these works the branching points in the network are determined ad hoc, which can lead to suboptimal task groupings.
```
2. soft share  

```
特点：each task is assigned its own set of parameters and a feature sharing mechanism handles the cross-task talk
问题：the size of the multi-task network tends to grow linearly with the number of tasks.
```

## **Decoder**  



## **Distilling Task Predictions**  

```
first employed a multi-task network to make initial task predictions, and then leveraged features from these initial predictions to further improve each task output – in a one-off or recursive manner.
(先得到初始预测，再任务内不同level以及任务间进行refine)
```
