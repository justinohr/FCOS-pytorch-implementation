## Experiment Environment
* Ubuntu 16.04
* Four GPUs (1080TI model is strongly recommended)
## How to train the model?
* python train_net.py --num-gpus 4 --config-file configs/R_50_1x.yaml
* If you are not with a four gpu environment, change batch size and learning rate according to [1].
## How to test the model?
* python train_net.py --config-file configs/R_50_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025
## Reference
* [1] P.Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, arXiv'17