

## Dataloader and lables

YOLO format [ class, x, y, w, h, k_x1, k_y1, visibility_flag1,  ..., k_x17, k_y17, visibility_flag17], and one person have one list including class, bounding box and 17 keypoints coordinate, sum 56 columns. 

k_x1 is not relative coordinate with box_x and box_y, it is absolute corrdinate.

All function in **create_dataloader** function to LoadImagesAndLabels.

```python
def cache_labels(self, path=Path('./labels.cache'), prefix='', kpt_label=False):
    ...
    if kpt_label:
        assert l.shape[1] == 56, 'labels require 56 columns each'
        assert (l[:, 5::3] <= 1).all(), 'non-normalized or out of bounds coordinate labels' # keypoint_x
        assert (l[:, 6::3] <= 1).all(), 'non-normalized or out of bounds coordinate labels' # keypoint_y
        # print("l shape", l.shape)
        kpts = np.zeros((l.shape[0], 39)) # 5+17x2=39
        for i in range(len(l)): # each person i
            kpt = np.delete(l[i,5:], np.arange(2, l.shape[1]-5, 3))  #remove the occlusion_paramater (visibility_flag) from the GT
            kpts[i] = np.hstack((l[i, :5], kpt)) # cat two list by horizontal 5+17x2 for GT 
            l = kpts
            assert l.shape[1] == 39, 'labels require 39 columns each after removing occlusion paramater'
        else:
            assert l.shape[1] == 5, 'labels require 5 columns each'
            assert (l[:, 1:5] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
    ... 
```
```python
 def __getitem__(self, index):
    ...
    # normalized for kpt x y
    if self.kpt_label:
        labels[:, 6::2] /= img.shape[0]  # normalized kpt heights 0-1
        labels[:, 5::2] /= img.shape[1] # normalized kpt width 0-1
    
    # flip up-down
    if self.kpt_label:
        labels[:, 6::2]= (1-labels[:, 6::2])*(labels[:, 6::2]!=0)
        
        
    # flip left-right
    if self.kpt_label:
        labels[:, 5::2] = (1 - labels[:, 5::2])*(labels[:, 5::2]!=0)
        labels[:, 5::2] = labels[:, 5::2][:, self.flip_index] # flip paramters according to flip index
        labels[:, 6::2] = labels[:, 6::2][:, self.flip_index]
```

```python
# Transform label coordinates
# augument for coorinate
def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0), kpt_label=False):
    ...
    # clip
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
    if kpt_label:
        xy_kpts = np.ones((n * 17, 3))
        xy_kpts[:, :2] = targets[:,5:].reshape(n*17, 2)  #num_kpt is hardcoded to 17
        xy_kpts = xy_kpts @ M.T # transform
        xy_kpts = (xy_kpts[:, :2] / xy_kpts[:, 2:3] if perspective else xy_kpts[:, :2]).reshape(n, 34)  # perspective rescale or affine
        xy_kpts[targets[:,5:]==0] = 0
        x_kpts = xy_kpts[:, list(range(0,34,2))]
        y_kpts = xy_kpts[:, list(range(1,34,2))]

        x_kpts[np.logical_or.reduce((x_kpts < 0, x_kpts > width, y_kpts < 0, y_kpts > height))] = 0
        y_kpts[np.logical_or.reduce((x_kpts < 0, x_kpts > width, y_kpts < 0, y_kpts > height))] = 0
        xy_kpts[:, list(range(0, 34, 2))] = x_kpts
        xy_kpts[:, list(range(1, 34, 2))] = y_kpts
        ...
```

```python
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0, kpt_label=False):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # it does the same operation as above for the key-points
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    if kpt_label:
        num_kpts = (x.shape[1]-4)//2
        for kpt in range(num_kpts):
            for kpt_instance in range(y.shape[0]):
                if y[kpt_instance, 2 * kpt + 4]!=0:
                    y[kpt_instance, 2*kpt+4] = w * y[kpt_instance, 2*kpt+4] + padw
                if y[kpt_instance, 2 * kpt + 1 + 4] !=0:
                    y[kpt_instance, 2*kpt+1+4] = h * y[kpt_instance, 2*kpt+1+4] + padh
    return y
```



## Detect Head framework 

```python
class Detect(nn.Module):
	def __init__(self, nc=80, anchors=(), nkpt=None, ch=(), inplace=True, dw_conv_kpt=False):
        ...
        self.nkpt = nkpt # num of keypoint
        self.dw_conv_kpt = dw_conv_kpt # subbranch of predict keypoint
        self.no_kpt = 3*self.nkpt ## number of outputs per anchor for keypoints
        ...
        if self.nkpt is not None: # predict keypoint [x,y,conf] module with different architecture
            if self.dw_conv_kpt: #keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                            nn.Sequential(DWConv(x, x, k=3), Conv(x,x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x,x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else: #keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)
                
    def forward(self, x):
    	...
    	for i in range(self.nl):
            if self.nkpt is None or self.nkpt==0:
                x[i] = self.m[i](x[i])
            else :
                x[i] = torch.cat((self.m[i](x[i]), self.m_kpt[i](x[i])), axis=1)
                
            # x (bs,num_anchor,feature_h,feature_w,5+nc)
            
            # only detect object and keypoint for person this class which num_class is 1 
            x_det = x[i][..., :6] # x,y,w,h,conf,cls
            x_kpt = x[i][..., 6:] # x_k,y_k,conf_k
            
            ...
            
            	y = x_det.sigmoid()
            	
            	# decode keypoint x and y with grid and stride, and conf of this keypoint
                x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()
                
                y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)
```



## Loss

```python
class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, kpt_label=False):
        super(ComputeLoss, self).__init__()
        self.kpt_label = kpt_label
		...

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCE_kptv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        
    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lkpt, lkptv = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], device=device) / 10.0
        tcls, tbox, tkpt, indices, anchors = self.build_targets(p, targets)  # targets
        ...
        for i, pi in enumerate(p):  # layer index, layer predictions
            ...
            # loss of keypoint regression and conf
            if self.kpt_label:
                #Direct kpt prediction
                pkpt_x = ps[:, 6::3] * 2. - 0.5
                pkpt_y = ps[:, 7::3] * 2. - 0.5
                pkpt_score = ps[:, 8::3]
                #mask
                kpt_mask = (tkpt[i][:, 0::2] != 0)
                lkptv += self.BCEcls(pkpt_score, kpt_mask.float()) 
                #l2 distance based loss
                #lkpt += (((pkpt-tkpt[i])*kpt_mask)**2).mean()  #Try to make this loss based on distance instead of ordinary difference
                #oks based loss
                d = (pkpt_x-tkpt[i][:,0::2])**2 + (pkpt_y-tkpt[i][:,1::2])**2 # x^2+y^2
                s = torch.prod(tbox[i][:,-2:], dim=1, keepdim=True) # w x h = scale of object
                kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
                lkpt += kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()
 
```

```python
def build_targets(self, p, targets):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = self.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, tkpt, indices, anch = [], [], [], [], []
    if self.kpt_label:
        # 7: image,class,grid_x,grid_y, grid_w,grid_h,anchor_idx
        # 34: 17 x 2
        gain = torch.ones(41, device=targets.device)  # normalized to gridspace gain
    else:
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ...
    
    for i in range(self.nl):
        anchors, shape = self.anchors[i], p[i].shape 
        if self.kpt_label:
            # w and h of grid of feature map 
            gain[2:40] = torch.tensor(p[i].shape)[19*[3, 2]]  # xyxy gain
        else:
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            
       	# Match targets to anchors
        # Define
        # Append
        if self.kpt_label:
            for kpt in range(self.nkpt):
                # keypoint offset with grid coordinate
                t[:, 6+2*kpt: 6+2*(kpt+1)][t[:,6+2*kpt: 6+2*(kpt+1)] !=0] -= gij[t[:,6+2*kpt: 6+2*(kpt+1)] !=0]
                tkpt.append(t[:, 6:-1])
```



## NMS

```python
                ...
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]
                ...
```

