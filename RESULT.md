# RESULT

## Patchmodel

start_lr =0.001,epoch = 5,比较最优的结果

| tile_shape | No_sch | SCH -STEP 20 |
| ---------- | ------ | ------------ |
| 128*16     | 97.71% | **98.99%**   |
| 64*64      | 97.30% | **97.48%**   |

## Noise 

noise_train.py

start_lr = 0.0001

std = 1 ,epoch = 10，计算ACC为随机加10次噪声， 取10个epoch中最优的

REPEAT在中间就进行均值处理

DECESION 对最终预测进行均值处理

|      | BASE   | REPEAT | DECESION   |
| ---- | ------ | ------ | ---------- |
| ACC  | 72.13% | 74.90% | **75.00%** |
