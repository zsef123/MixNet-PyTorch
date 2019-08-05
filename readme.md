# Mixnet

A PyTorch implementation of `MixNet: Mixed Depthwise Convolutional Kernels.`


### [[arxiv]](https://arxiv.org/abs/1907.09595) [[Official TF Repo]](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet)

<hr>

## Acknowledge

Now EMA is running on CPU. So It slower than normal runner.

If you running on GPU, then change these lines [init](ema_runner.py#23), [update_ema](ema_runner.py#96)

<hr>

## How to use:

```
python3 main.py -h
usage: main.py [-h] --save_dir SAVE_DIR [--root ROOT] [--gpus GPUS]
               [--num_workers NUM_WORKERS] [--model {mixs}] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--test] [--ema_decay EMA_DECAY]
               [--optim {rmsprop,adam}] [--lr LR] [--beta [BETA [BETA ...]]]
               [--momentum MOMENTUM] [--eps EPS] [--decay DECAY]
               [--scheduler {exp,cosine,none}]

Pytorch Mixnet

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory name to save the model
  --root ROOT           The Directory of data path.
  --gpus GPUS           Select GPU Numbers | 0,1,2,3 |
  --num_workers NUM_WORKERS
                        Select CPU Number workers
  --model {mixs}        The type of mixnet.
  --epoch EPOCH         The number of epochs
  --batch_size BATCH_SIZE
                        The size of batch
  --test                Only Test
  --ema_decay EMA_DECAY
                        Exponential Moving Average Term
  --optim {rmsprop,adam}
  --lr LR               Base learning rate when train batch size is 256.
  --beta [BETA [BETA ...]]
  --momentum MOMENTUM
  --eps EPS
  --decay DECAY
  --scheduler {exp,cosine,none}
                        Learning rate scheduler type
```
