# MonoScene   
流程:    
1. 2D Unet 提取特征；   
2. Flosp将2D特征转换到3D特征；   
3. 3D Unet提取特征（3D CRP）；   
4. 预测头预测是否占用和语义类别；   
5. Loss：在分类交叉熵loss基础上，增加KL散度的分布约束。   
![](files\_j)    
   
Flosp：将3D空间体素中心投影到2D的不同scale特征图上，然后基于投影后的2D点对多尺度2D特征进行采样，并对多尺度采样特征进行加权求和或者平均以表示3D特征。   
![](files\_5)    
3D CRP: 计算N个voxel之间相互的关系矩阵类似于attention map来增加上下文信息，每对voxel之间有4种关系（one is free or occupied, their semantics is similar or different.），即有4种关系各自的attention map，基于这4个不同的关系矩阵与特征图作矩阵相乘，并concat作为最终特征。为了降低度，将N个voxel缩减为k个不重叠的supervoxel（类似于聚类中心降维的思路），而4种关系也简化为多分类预测用loss约束。   
![](files\_h)    
