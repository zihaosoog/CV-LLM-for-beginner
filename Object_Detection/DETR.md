## DETR : End-to-End Object Detection with Transformers
<b>Abstract</b>: bipartite matching; transformer encoder-decoder architecture model objects relation; better on large objects

<b>Introduction</b>:
1. viewing object detection as a direct setprediction problem by self-attention mechanisms
2. removing duplicate predictions<b> (no NMS)</b>
3. dropping spatial anchors
4. autoregressive decoding: 
   last time output as next time input. In other words, predicting results <b>one by one (RNNs)</b>
5. non-autoregressive decode: 
   fedding a set of inputs at the same time, then <b>decoding N objects in parallel (Transformer)  </b>
6. extra-long training schedule and extra-long training schedule
   
<b>Related work</b>:

Set-based loss: <b> Non-unique assignment rules</b>   (GT<-->prediction) + <b> NMS</b> (replace by attention)
   
### <b>DETR</b>:
<b>Architecture</b>

![Fig2](https://pdf.cdn.readpaper.com/parsed/fetch_target/fe95d3164d4c868cd30406f698068bdf_6_Figure_2.png)

code simple version:
    
    import torch
    from torch import nn
    from torchvision.models import resnet50
    
    class DETR(nn.Module):
    
        def __init__(self, num_classes, hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers):
            super().__init__()
            # We take only convolutional layers from ResNet-50 model
            self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
            self.conv = nn.Conv2d(2048, hidden_dim, 1)
            self.transformer = nn.Transformer(hidden_dim, nheads,num_encoder_layers, num_decoder_layers)
            self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
            self.linear_bbox = nn.Linear(hidden_dim, 4)

            # output positional encodings (object queries)
            # 一张图片最多检测100个物体
            self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
            
            # spatial positional encodings
            # note that in baseline DETR we use sine positional encodings
            # 编码器的位置编码，对特征图行和列分别进行位置编码，后面会将两者拼接起来故维度为一半
            # 特征图尺寸不超过50*50  50个行，50个列最多
            self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
    
        def forward(self, inputs):
            # input (3, H0, W0)
            x = self.backbone(inputs)
            # x (C=2048, H0/32, W0/32)
            h = self.conv(x)
            # 1x1 conv -> h (d, H, W)
            H, W = h.shape[-2:]
            # flatten(0, 1) -> (d,HW)
            pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1), # [H,W,hidden_dim//2]
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(1)
            h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
            self.query_pos.unsqueeze(1))
            # FFNs (linear)
            return self.linear_class(h), self.linear_bbox(h).sigmoid()
    
    detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
    detr.eval()
    inputs = torch.randn(1, 3, 800, 1200)
    logits, bboxes = detr(inputs)

<b>Loss</b>

N >> the number of objects in the image, y [class, x_c, y_c, w, h] : a set of size N padded with ∅ (no object)

![bipartite_matching](https://img-blog.csdnimg.cn/20200611091538612.png)

loss: probaility of class loss + predicted box loss

![图片](https://user-images.githubusercontent.com/67272893/149099475-2a0f1cda-e8be-48f7-af37-9171de7205d3.png)

class imbalance: 
1. DETR: when ci is empty, down weight the log()
2. Faster R-CNN:  training procedure balances positive/negative proposals by subsampling

![Hungarian loss](https://img-blog.csdnimg.cn/20200611091601304.png)

![box loss](https://img-blog.csdnimg.cn/20200611091615382.png)



<b> </b>

References:
> [自回归解码与非自回归解码简介](https://zhuanlan.zhihu.com/p/427311331)

> [DETR源码阅读](https://blog.csdn.net/qq_43173239/article/details/114208214#t5)

> [Loss 思路](https://blog.csdn.net/lgzlgz3102/article/details/117794551)
