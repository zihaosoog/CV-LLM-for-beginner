## 3D感知问题具体指什么

多个视角的图像分别检测目标无法完整感知周围目标，例如视角图像间有重叠区域，单个目标会重复检出；目标太大单个视角无法覆盖则任意单个视角都无法完整检出。

## 感知融合分为哪几种

后融合：先独立感知目标，然后再将结构化信息以目标级进行时序和多传感器融合
缺点：对结构化信息融合存在信息损耗，融合需要先验假设如目标运动模型等超参数限制

中融合：先对各个视角图片提取feat，然后在统一特征空间融合各视角feat，最后感知预测

## 中融合分为哪几种

1. 密集感知（BEV生成密集特征进行感知）

   2D到3D：
   思路：基于LSS，通过提升深度估计的精度以提升感知性能。
   示例：如BEVDepth用点云监督深度估计；BEVStereo把时序序列帧当作双目图像进行深度估计；VideoBEV采用recurrent时序融合方式，将当前帧的BEV特征和当前帧的长时序memory作融合（融合：拼接再conv输出predict），再将本时刻融合后的BEV特征（作为下一帧的长时序memory）传递到下一帧作融合。

   3D到2D：
   思路：将3D空间中的点映射回各个视角下的2D图中，并采样对应的2D feat作为3D空间特征表示。
   示例：基于IPM 的BEV方法将所有物体假设都在地面高度，即BEV grid的3D点都把高度置为0再投影到2D视角图中，获得BEV grid 的特征；BEVFormer将BEV grid的z轴分为N块从而得到N个Z轴不同的3D点，分别映射到2D采样img feat，获得更加丰富的BEV grid特征，同时对cat后时序相邻帧feat做采样获得时序信息；（如果说刚刚的IPM是一个BEV Grid采样一个点，BEVFormer就是一个Grid采样了非常多的点。）

2. 稀疏感知
   DETR3D只采用稀疏的Query进行逐步迭代更新信息完成感知预测，具体来说将3D query通过内外参转换到各个视角的2D feat中做特征采样，融合采样特征来更新3D query：PETR构建3D坐标网格（相机截头体空间）并通过内外参转换到3D真实坐标系的坐标网格，将3D网格坐标和2D feat相结合获得网格所有的3D feat，将3D query与3D feat进行交互更新query；
   DETR3D是稀疏Query加上稀疏的特征交互；PETR则是稀疏的Query加上密集的特征交互；PETR-V2 和StreamPETR 则分别引入了两帧的时序和Recurrent的时序形式。

   Sparse4D

   1. backbone采用多视角多时刻图片作为输入，提取多视角多时刻各自的多尺度特征图，作为encoder的输出

   2. Decoder基于DAB-DETR思想，初始化输入两部分，包括3D Anchor（回归大小位置角度等）以及instance feature（分类预测以及关键点refine）。

      对于**3D Anchor的迭代refine**，基于3D Anchor信息，预测固定关键点（立体anchor box每个面的中心以及立体的体中心）+可学习关键点（立体anchor box的其他点位，通过instance feature的MLP变换而来）。**为了结合时序信息**，该算法结合自车运动速度信息，将t0当前时刻的*3D anchor*转换推算出其他时刻的*3D anchor’*，相当于把当前帧的一系列关键点投影到了每一个历史帧上，结合当前帧和历史帧的3D关键点，就获得了每个实例的4D的关键点。基于后续融合的instance feature进行anchor相应的offset预测等refine的操作。

      对于**instance feature的迭代refine**，将4D anchor（即多时刻的3Danchor）通过内外参投影到多视角多尺度多时刻的特征图上作特征采样，在视角和尺度两个层面进行分组通道加权融合，在时序层面上采用RNN进行顺序融合。
      为了减少不同3D点映射到相同2D点的问题，最后通过深度估计的方式对instance feature作进一步的约束（加权），提高模型收敛速度。
   
   Sparse4D-V2
   
    1. encoder与v1相同
   
    2. Decoder分为非时序层（1层）和时序层（5层）。
   
       首先非时序层输入初始化的 anchor query和 instance feature（类似于content query），经过多视角以及多尺度特征融合，在作进一步特征提取，经过refine输出 topK的前景score和相应anchor（类似于ROI的前背景2分类筛选高质量query）。
   
       其次将上述topK的预测`instance_t`结果作为input的一部分，同时将上一帧t-1的instance投影到当前t帧（包括两部分，instance feature保持不变，anchor通过速度等ego信息投影到t帧）得到`instance_t-1->t`，将`instance_t`和`instance_t-1->t`两部分混合进行cross-attention/self-attention，再进行FFN，预测输出并选取topK的结果作为t+1帧的时序信息进行下一次的时序融合。



## 为什么车端BEV实现远距感知比较差

常用的nuScenes数据集中，一般感知范围会设置为长宽 [-50m, +50m] 的方形区域，但在实际场景中，我们通常会需要达到单向100米，甚至200米的感知距离。
如果说我们想要保持BEV Grid的分辨率不变，那么就需要去增加BEV特征图的尺寸，这会使得端上的计算负担和带宽负担都非常重。
如果要保持BEV特征图的尺寸不变，就需要更加粗粒度的BEV Grid，那么它的感知精度就会下降。

## 自动驾驶中坐标系分为哪几大类

3类，世界坐标系（如地心坐标系），车身坐标系，各类传感器坐标系（包括相机坐标系，lidar坐标系等）