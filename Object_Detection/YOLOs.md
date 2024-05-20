## **YOLO v1**

![yolo](https://pic4.zhimg.com/v2-aad10d0978fe7bc62704a767eabd0b54_b.jpg)

    1）图片输入：3×448×448 --> 24个卷积层以及2个全连接层 --> 特征输出：30×7×7 （相当于将原图划分为7×7个方格）
    2）特征图点映射回原图，每个中心点设置 2 个不同的box，一张原图中共有7×7×2个box，但对于一个中心点来说，其所有box只能保留1个IoU最大的，即每个方格只能预测1个物体
    3）30通道 ：
    20类概率（选用2个box中高IoU的box，其对应的softmax类别输出，无背景这一类别）；
    2个置信度（2个box，若包含物体则值为对应IoU/不包含物体则值为0）；
    2×4（坐标宽高）；
    4）正样本：物体中心落在方格内，并取该方格对应的高IoU-box为正样本，除此之外都为负样本
    5）损失函数计算：3类特征都利用均方误差进行回归，负样本只计算置信度损失
    6）YOLO-v1 物体类别与置信度分开，而Faster RCNN预测类别的过程包含了置信度的预测 
    7) 测试时,利用阈值对(class-score乘以box-confidence)进行过滤,再进行NMS
    
    
    
## **YOLO v2**

[YOLOv2 & YOLO9000 解析参考](https://zhuanlan.zhihu.com/p/25052190)

    1）DarkNet-19网络输出特征：13×13×125，每个区域预测5个box，每个box包含20类分类概率+4个偏移量预测值+1个置信度预测
       Direct Location prediction：利用预测偏移与当前grid cell坐标关系计算输出预测框位置大小
    2）修改网络输入尺寸：由448×448改为416，YOLOv2的卷积层下采样率为32，因此输入尺寸变为416,输出尺寸为13×13，奇数尺寸使特征图只有一个中心。物品（特别是大的物品）更有可能出现在图像中心。
    3）anchor先验框生成：利用kmeans聚类对训练集中的bbox归类得到k(k=5)个不同尺寸的先验框，两个bbox之间距离定义为：1-IoU(bbox1,bbox2)，相比Faster RCNN中人为设计9个不同尺寸的anchor
    4）passthrough layer：类似与残差结构，将浅层特征隔行采样堆叠完成降维，与深层特征融合，提高小目标检测效果
    5）多尺度训练：每10个epoch变换输入的尺寸，由于下采样率为32，输入尺寸集合{320，352，384,...,608}
    6）高分辨率预训练：
       yolo v1 预训练网络（分类）时：以224×224输入；接着fine-tune网络（检测）时：448×448输入；转换不好
       yolo v2 预训练网络（分类）时，先以224×224输入图像训练160 epochs，然后再以448×448输入训练10 epochs；然后fine-tune网络（检测）时：448×448输入
    7）YOLO9000：联合训练：将Imagenet标签结构层次用WordTree表示，再将COCO（检测）与Imagenet（分类）标签融合为一个新的WordTree
       具体来说：During training we mix images from both detection and classification datasets. 
       When our network sees an image labelled for detection we can backpropagate based on the full YOLOv2 loss function. 
       When it sees a classification image we only backpropagate loss from the classification specific parts of the architecture.
    8）多层分类：类似于决策树，每次分类只对同一层的节点softmax，每个节点概率就是从根节点到此节点路径中各概率乘积
       Using this joint training, YOLO9000 learns to find objects in images using the detection data in COCO and it learns to classify a wide variety of these objects using data from ImageNet.


## **YOLO v3**

[YOLO v3图解](https://zhuanlan.zhihu.com/p/345073218)

    1）DarkNet-53：用卷积步长代替pooling完成下采样，采用残差块
    2）多尺度特征预测：类似与FPN，3种尺寸的特征图输出：13×13，26×26，52×52
       同样通过Kmeans聚类整个训练集的bbox，得到9个大小不同的anchor，排序分配每个尺寸特征图有3种anchor，
       每种特征图对应不同原图划分，特征图尺寸多大划分多少个格子，划分的每个格中心点有3个anchor
       大尺寸特征图 --> 感受野小，适合小目标 --> 分配小尺寸anchor
       预测输出（batchsize，anchor_num, 方格数，方格数，（1+4+num_class））
    3）深层特征（尺寸小）经过上采样（插值方法）与浅层特征通道拼接
    4）预测框与目标一一对应，但使用多个独立sigmoid代替softmax，使得1个anchor可实现多标签预测（类别解耦），比如同时得到“女人”和“人”2个标签
    5）使用交叉熵损失计算类别概率，置信度（框内有物体的概率）采用对数求和的损失
    
 ### code note
 
    class Model: m = eval(m) if isinstance(m, str) else m  # eval strings
    
[eval(m)将字符串变为所import的类](https://blog.csdn.net/alittlebai1/article/details/118442444)

    class Detect: def _make_grid(self, nx=20, ny=20, i=0)
    
[构建网格全过程讲解](https://www.cnblogs.com/yanghailin/p/15338637.html)
    
## **YOLO v4**

![image](https://user-images.githubusercontent.com/67272893/142413859-71998620-697a-4747-885d-dfd8e1d0d3f5.png)


### Object detection models
> 一般目标检测网络结构

- Detector = Backbone + Neck + Head

> Backbone: 在Imagenet上预训练

    GPU: VGG,Resnet,ResnetXt,DenseNet
    CPU: SqueezeNet,MobileNet,ShuffleNet
    
> Head: 预测目标类别和bbox位置

    two-stage: fast RCNN,faster RCNN,R-FCN,Libra RCNN,RepPoints(anchor-free)
    one-stage: YOLO,SSD,RetinaNet,CenterNet(anchor-free),CornerNet(anchor-free),FCOS(anchor-free)
    
> Neck: 位于backbone和head之间，收集不同阶段特征图，自顶向下或者自底向上

    Path-aggregation blocks: FPN,PAN,BiFPN,NAS-FPN
    Additional blocks: SPP,ASPP,RFB,SAM
    
### Bag of freebies: training strategy

> Data augmentation：增加输入多样性，使网络更鲁棒

    1) 光学变换：调整亮度，对比度，噪声等
    2）几何变换：随机尺寸变换，剪切，翻转，旋转等
    3）在原图或特征图上随即擦除，不同图像局部/全局混合（标签随之变化），风格迁移
       
> Semantic distribution bias：不同类别，样本数量不均衡

    two-stage detector: hard negative example mining, online hard example mining
    one-stage detector: focal loss
    
> One hot hard label: 无法表达不同类别之间的关联

    soft label（或者采用知识蒸馏细分标签）
    
> BBox 回归的目标函数

    利用MSE估计BBox的坐标点offset以及长宽offset
    IoU loss：尺度不变性，L1和L2损失不具有尺度不变性
    GIou loss: include shape and orientation
    DIoU loss: 考虑目标与anchor之间中心点距离
    CIoU loss: 考虑覆盖区域，中心点距离，长宽比
    
### Bag of specials

> plugin modules

    enhance receptive field:    
        ASSP: 在SSP基础上，最大池化改用空洞卷积，采用若干个卷积核尺寸为3×3，步长为1，dilated ratio(扩张率)为k的空洞卷积
        RFB: dilated convolutions of k×k kernel, dilated ratio equals to k, and stride equals to 1
    attention: SE(channel-wise attention), SAM(point-wise attention)
    feature integration: skip connection, hyper-column, e.g. SFAM(通道加权)，ASFF(点加权)，BiFPN(不同尺度加权)
    activation function: 解决梯度消失：LReLU, PReLU, ReLU6, Scaled Exponential Linear Unit (SELU), Swish, hard-Swish, and Mish, etc.

> post-processing methods

    NMS，DIoU 

### Method

> architecture selection

    the input network resolution, the convolutional layer number, the parameter number, and the number of layer outputs
    increasing the receptive field and parameter aggregation

> improvements

![image](https://user-images.githubusercontent.com/67272893/143015453-9901ae09-73e0-4982-a69c-7d1dea82eaca.png)

    数据增强：Mosaic(4张图mix),SAT(生成虚假图片对抗训练)
    超参数：利用遗传算法选择超参数
    其他：SAM(注意力机制中score生成方式稍作改变)，PAN(自下向上过程中连接方式由+改为*)，CmBN(均值和方差每个batch内阶梯累计计算，其他与BN相同)
    
    

