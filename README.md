
# AE TextSpotter

## Introduction
This is the official implementation of AE TextSpotter, which introduces linguistic information to eliminate the ambiguity in text detection.
This code is based on [MMDetection v1.0rc1](https://github.com/open-mmlab/mmdetection/tree/v1.0rc1).

![demo image](demo/demo.jpg)

## Recommended environment
```
Python 3.6+
Pytorch 1.1.0
torchvision 0.2.1
pytorch_transformers 1.1.0
mmcv 0.2.13
Polygon3
opencv-python 4.4.0
```

## Install
Please refer to [MMDetection v1.0rc1](https://github.com/open-mmlab/mmdetection/tree/v1.0rc1) for installation.

## Preparing data
Step1: Downloading dataset from [ICDAR 2019 ReCTS](https://rrc.cvc.uab.es/?ch=12).

Step2: The root of data/ReCTS should be:
```
data/ReCTS/
├── train
│   ├── img
│   ├── gt
├── test
│   ├── img
```

## Training
Step1:
```shell script
tools/rects_dist_train.sh local_configs/rects_ae_textspotter_r50_1x.py 8
```
Step2:
```shell script
tools/rects_dist_train.sh local_configs/rects_ae_textspotter_lm_r50_1x.py 8
```
Step3:
Download and unzip [bert-base-chinese.zip](https://drive.google.com/file/d/1ufa3gl95Fpdd6duUqu6Fb3m85sbDRjpt/view?usp=sharing) in the root of this repository.
```shell script
unzip bert-base-chinese.zip
```

## Test
```shell script
tools/rects_dist_test.sh local_configs/rects_ae_textspotter_lm_r50_1x.py work_dirs/rects_ae_textspotter_lm_r50_1x/latest.pth 8 --json_out results.json
```

## Visualization
```shell script
python tools/rects_test.py local_configs/rects_ae_textspotter_lm_r50_1x.py work_dirs/rects_ae_textspotter_lm_r50_1x/latest.pth --show
```

## Evaluation
The training list, validation list, and evaluation script of this code come from [TDA-ReCTS](https://github.com/whai362/TDA-ReCTS)
```shell script
python tools/rects_eval.py
```
The output of the evaluation script should be:
```shell script
[Best F-Measure] p: 84.94, r: 78.10, f: 81.37, 1-ned: 51.02, best_score_th: 0.569
[Best 1-NED]     p: 86.68, r: 76.09, f: 81.04, 1-ned: 51.51, best_score_th: 0.626
```

## Results and Models
[TDA-ReCTS](https://github.com/whai362/TDA-ReCTS)

| Method | Precision (%) | Recall (%) | F-measure (%) | 1-NED (%) | Model |
| - | - | - | - | - | - |
| AE TextSpotter | 84.94 | 78.10 | 81.37 | 51.51 | [Google Drive](https://drive.google.com/file/d/1VcYygCZYxZCY2ruuVOppQRVIds0Blb0a/view?usp=sharing) |

[ICDAR 2019 ReCTS](https://rrc.cvc.uab.es/?ch=12)

Todo.


## License
This project is released under the [Apache 2.0 license](LICENSE).

## Citation
If you use this work in your research, please cite us.
```
@inproceedings{wenhai2020ae,
  title={AE TextSpotter: Learning Visual and Linguistic Representation for Ambiguous Text Spotting},
  author={Wang, Wenhai and Liu, Xuebo and Ji, Xiaozhong and Xie, Enze and Liang, Ding and Yang, ZhiBo and Lu, Tong and Shen, Chunhua and Luo, Ping},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Other Projects:
PAN (ICCV 2019): https://github.com/whai362/pan_pp.pytorch

PSENet (CVPR 2019): https://github.com/whai362/PSENet