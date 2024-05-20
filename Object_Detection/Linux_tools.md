> 指定GPU运行程序
    
    CUDA_VISIBLE_DEVICES=0 nohup python detect.py >cmdout.out 2>&1 &

> 切换端口

     ssh 192.168.8.206
     
> 查看GPU使用情况

    nvidia-smi

> 刷新GPU使用

    watch --color gpustat --color   # ctrl+z 退出

> ptvsd

    python -m ptvsd --host 222.131.60.5 --port 205 --wait test_ptvsd.py

    
>  新建会话

    tmux new -s <session-name>
    
> tmux 接入会话

    tmux attach -t <session-name>

> 激活conda虚拟环境

    source activate objdet


> 假设已有环境名为A，需要生成的环境名为B：

    conda create -n B --clone A
    
> 在vscode中更改anaconda运行环境

    快捷键ctrl+p，在弹出的框框中，输入 >select interpreter 来选择相应的Anaconda环境即可。

> mmdetection single GPU test

    python tools/test.py configs/faster_rcnn_r50_fpn_1x.py work_dirs/faster_rcnn_r50_fpn_1x/epoch1.pth --eval bbox --show --options "classwise=True"
