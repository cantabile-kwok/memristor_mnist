# 基础部分

## noise scale

首先用batch size=16，两层卷积+两层FC的CNN，测了一下不同噪声对于acc的影响。图画在![std_acc.png](figs/std_acc.png)。

所以之后用std=1.2应该比较合适。注意这里的CNN model acc是99.15%。

## batch size

固定std=1.2，model=MLP or CNN，lr=0.001。