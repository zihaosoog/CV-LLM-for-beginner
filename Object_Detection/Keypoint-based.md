## **CornerNet**

> corner pooling（以左上角为例）
 
    两个不同特征图上不同方向进行maxpooling
    大小：C×H×W，C表示物体类别；HW上每个位置表示每个点是否为角点的概率

![image](https://user-images.githubusercontent.com/67272893/148005099-f258f8b0-2efd-4099-8fee-4ee7f62d20ee.png)

    maxpooling：找到走过的路中包含当前位置最大的值

![image](https://user-images.githubusercontent.com/67272893/147025619-c917e74e-e037-4eb5-be36-364eadf73e0f.png)

> 预测输出

    heatmap：成为角点的概率
    embedding：计算两角点embedding之间的距离，判断是否为一组角点
    offset：修正的偏移量

![image](https://user-images.githubusercontent.com/67272893/147028761-5d39888a-0b33-4263-84da-59125109ef4d.png)

> 网络结构

    Hourglass Networks：先对图像下采样再上采样恢复尺寸大小获得全局信息，同时使用shortcut补全细节信息
    
![image](https://user-images.githubusercontent.com/67272893/147030965-e4e376d3-1134-44ce-a320-0b7f86ba7d7e.png)


## **CenterNet：Keypoint Triplets**

>  Key words: a triplet of keypoints; cascade corner pooling and center pooling

    1. 选择top-k bbox以及中心点，映射回原图
    2. 查看中心点是否在bbox的中心方格内，并查看中心点与bbox标签是否一致

![image](https://user-images.githubusercontent.com/67272893/147032059-e2a251f2-5645-4d2c-ac31-2661d5370df1.png)

> Center pooling: 分别找到在水平和竖直方向的最大值，将他们综合得到中心点

![image](https://user-images.githubusercontent.com/67272893/147051112-dad4328c-5d77-4526-8dc0-1fa40a06b274.png)

> Cascade corner pooling: 从角点沿着边界找最大值，找到后，再向框内部寻找最大值，综合边缘与内部

![image](https://user-images.githubusercontent.com/67272893/147051219-2ef1f71b-6e70-44fb-b222-c7d1d5b53880.png)

## **Objects as Points**

[Centernet笔记参考](https://zhuanlan.zhihu.com/p/66048276)

    与 anchor-based one stage 相比，有以下不同：
    1.“anchor”只关于位置，不关于box overlap
    2.每个目标只有一个bbox，不需要NMS
    3.lager output resolution(output stride small)
    
    网络：
    Hourglass network(like conrner net)；
    Resnet；
    DLA；
    
    预测流程：
    网络输出接三个head，分别输出 类别预测即heatmap，坐标offset，长宽，即下采样后的每个点对应输出C(calss num)+4(offset,w,h)个输出
    其中采用maxpooling选取center point，用loss_off回归offset,用回归的offset矫正center point，再根据长宽绘制bbox
    
    loss：
    Focal loss: 解决分类问题中类别不平衡，分类难度差异大,难样本惩罚大即损失大
    loss_off回归offset，loss_size回归bbox大小
    
    
