# SDR
This is the Python implementation of our paper "SDR: Debug repositioning based on self supervised graph learning".

>Sichen Jin, Yijia Zhang*, Huimin Yu, and Mingyu Lu

## Environment Requirement

The code runs well under python 3.7. The required packages are as follows:

- pytorch == 1.7.1
- numpy == 1.20.3
- scipy == 1.7.1

## Quick Start
**Firstly**, you need to download the corresponding dataset, which can be downloaded from [here](https://github.com/luoyunan/DTINet) or [Baidu Cloud](https://pan.baidu.com/s/1Z82WaLBblt1_BjjMKWYzEw?pwd=k9fy). The extraction code is k9fy. Then place the downloaded dataset file in the dataset folder.


**Secondly**, change the value of variable *root_dir* and *data_dir* in *main.py*


Some important hyperparameters:

### DeepDR dataset
```
aug_type=ED
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.1
ssl_ratio=0.1
ssl_temp=0.2
```

### DTI dataset
```
aug_type=ED
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.5
ssl_ratio=0.1
ssl_temp=0.2
```

### CDataset dataset
```
aug_type=ED
reg=1e-3
embed_size=64
n_layers=3
ssl_reg=0.02
ssl_ratio=0.4
ssl_temp=0.5
```

**Finally**, run [main.py](./main.py) in IDE or with command line:

### DeepDR dataset
```bash
python main.py --recommender=SDR --dataset=DeepDR --aug_type=ED --reg=1e-4 --n_layers=3 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
```

### DTI dataset
```bash
python main.py --recommender=SDR --dataset=DTI --aug_type=ED --reg=1e-4 --n_layers=3 --ssl_reg=0.5 --ssl_ratio=0.1 --ssl_temp=0.2
```

### CDataset dataset
```bash
python main.py --recommender=SDR --dataset=CDataset --aug_type=ED --reg=1e-3 --n_layers=3 --ssl_reg=0.02 --ssl_ratio=0.4 --ssl_temp=0.5
```
