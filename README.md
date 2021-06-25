# IPT_core
The core part of Image Processing Transformer (CVPR2021) in Pytorch (Pytorch 1.8).

The third party implmentation of the core part (Transformer) of IPT.

Key: Patch-based Multi-head Self-Attention.

IPT paper: [Pre-trained Image Processing Transformer](https://arxiv.org/pdf/2012.00364.pdf)

![image](https://user-images.githubusercontent.com/30970296/123435135-ce553700-d5ff-11eb-9292-66ed057adde1.png)

FLOPs & Params analysis (with Non-local Neural Network):

Patch_size: 3x3

![image](https://user-images.githubusercontent.com/30970296/123436537-4d973a80-d601-11eb-8fa2-6bbc1687ca5e.png)

![image](https://user-images.githubusercontent.com/30970296/123436258-09a43580-d601-11eb-8221-7a8dd23c4828.png)

| Input size                                     | (64, 30, 30) | (64, 60, 60) | (64, 120, 120) | Params. |
|------------------------------------------------|:------------:|:------------:|:--------------:|:-------:|
| Non-local Neural Network                       |    0.0633    |    0.9237    |     14.4258    |  8.35K  |
| Patch_based_Multihead_attention (IPT, p_s=3x3) |    0.0449    |    0.3202    |     3.5311     |  1.33M  |
| Patch_based_Encoder_layer (IPT, p_s=3x3)       |    0.3109    |    1.3842    |     7.7871     |  3.99M  |
| Patch_based_Decoder_layer (IPT, p_s=3x3)       |    0.3561    |    1.7056    |     11.3229    |  5.32M  |
