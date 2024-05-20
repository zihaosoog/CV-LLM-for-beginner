### **RCNN**

![RCNN](https://img-blog.csdn.net/20160308074838490?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

        1) 候选区生成（region proposals）：采用selective search，即过分割，再根据比如颜色以及纹理梯度直方图等属性，将相邻且相似的区域合并；
        2）CNN提取特征：采用AlexNet，输入裁剪得到的每个候选区并reshape为固定大小，输出每个候选区的4096维特征，去掉预训练模型中的分类层，训练新的分类器；
        3）SVM分类器：每一类目标对应一个二分类的SVM分类器（是或否），其中GT作为正样本，IoU小于0.3的候选区作为负样本；
        4）NMS非极大抑制：对于一个目标有多个候选框，根据SVM计算得分与候选框之间的IoU保留一个候选框作为bbox；
        5）位置精修：将bbox的中心点坐标以及长宽与GT之间作回归，训练线性回归参数；
        6）训练过程：预训练模型去除FC -> 在21类情况下微调模型参数去掉分类器 -> 提取特征 -> 训练SVMs -> 回归
        7）实时性：47s/image；候选区域提取+特征提取（13s/image-GPU）；分类+精修（10s）
        
        
### **SSP**

![SSP](https://img-blog.csdnimg.cn/20190212151939704.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZlbmdiaW5nY2h1bg==,size_16,color_FFFFFF,t_70)

        1）根据调整池化窗口大小以及步长，使得输出特征图尺寸相同，而每个输入图片都输出3种固定大小特征图，即空间金字塔池化
        2）Sub-Sampling ratio（S）:输入图片中的 region proposal 尺寸以及坐标 -> 乘以S -> 得到特征图中region proposal的尺寸以及位置坐标
        3）因为FC层需要输入固定大小的特征图，所以SPP层添加在FC之前一层
        
        
### **Fast RCNN**

![Fast RCNN](https://img-blog.csdnimg.cn/20190212152138464.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZlbmdiaW5nY2h1bg==,size_16,color_FFFFFF,t_70)

       1）预训练VGG-16，用 RoI pooling 层代替原来的池化层，去掉FC层，取而代之采用两个并行的FC层用来分类和回归
       2）输入为整张图片和 region proposal（采用selective search），将两者映射到特征图中，其中 region proposal 映射方式与 SSP 中相同
       3）RoI pooling layer：类似于SSP中的空间金字塔池化层，这里采用1种固定大小的特征图
       4）将RoI的固定大小的特征图输入FC层进一步提取特征（采用奇异值分解加速计算）
       5）通过并行两支路FC层，用softmax分类（背景+目标类别数），对bbox的（x,y,h,w）回归
       6）多任务损失函数：分类loss即正确分类概率的负对数，回归loss即smoothL1(预测与真实线性变换参数差)，两个损失函数加权，加权使得负样本不参加回归损失
       7）采样方式：RCNN和SSP对所有图像的所有候选区域均匀采样（R个候选区），每个mini-batch包含不同图像的候选区域，不能共享卷积
                   Fast RCNN 先采样图片（N张），再对每个图像采样（R/N个候选区），同一图像内可共享计算与内存
       8）实时性：0.5 FPS
                            
                            
### **Faster RCNN**

[全过程详细解读](https://zhuanlan.zhihu.com/p/31426458)

        1）如何将Anchor从特征图映射回原图：
           特征图中单位方格中心点 -> 乘以S采样率 -> 恢复到原图中 即整个感受野区域的中心点 -> 根据设定的basesize以及ratio以该点为中心 生成该点的Anchors
        2）卷积层提取特征：采用 kernel_size=3, padding=1,使得输入和输出尺寸不变
        3）RPN（生成 proposal region）：
           对特征进行3×3卷积 -> 分两支路（分类和回归）-> 分类: 特征维度 输入softmax：2×9×H×W (两类×anchor个数×特征图高×特征图宽)，reshape变换为了符合分类输入要求
                                                   -> 回归:真实偏移量与缩放 与 线性变换预测的偏移量 进行回归 输出：4×9×H×W
                                                   -> proposal layer：根据softmax scores选取 用预测偏移量 修正的 positive anchors，剔除不合理的anchor并NMS减少重叠
        4）RoI pooling与classification：与RCNN相同，两分支分类和回归损失（其实是第二次分类回归，RPN中第一次）
        5）训练：end to end, like a single net,速度快
