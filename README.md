### Official implementation of paper "Prior-Enhanced Metric Learning for Few-Shot Recognition of Counterfeit Cigarettes Using Packaging Information"

#### 1. Environment 

```txt
matplotlib==3.6.2
opencv-python==4.9.0.80
Pillow==9.3.0
PyYAML==6.0.1
scikit-learn==1.4.1.post1
torch==1.13.1
```

#### 2. How to train/test

```shell
python main.py --cfgs ./configs/base.yaml --phase train
python main.py --cfgs ./configs/base.yaml --phase test
```

You may need to change the `configs/base.yaml` according to your dataset.

#### 3. Dataset Preparation

Data should be organized as follows:

```txt
brand_name/
├── train/
│   ├── region0/
│   │   ├── real/
│   │   │   ├── img1
│   │   │   └── img2
│   │   └── fake/
│   └── region1/
└── test/
```



![img](http://cdn.lisan.fun/img/image-20240713165022388.png)

#### 4. Inference

We release some test samples of 1916_scan and 1916_camera, as well as the model weights. You can try it as follows:

```shell
python inference.py
```

#### 5. Data availability

To safeguard against counterfeiters using our dataset to improve their forgery methods, we will not disclose the collected cigarette data or the anti-counterfeiting regions linked to each cigarette specification. For legitimate inquiries, please contact [zhouqian@whu.edu.cn](mailto:zhouqian@whu.edu.cn) and enter into a confidentiality agreement.

#### 6. Cite

If you find the repo useful to your work, please cite our paper.

