## Pytorch Lightning

> pl 是对Pytorch的抽象封装。快速上手教程可以参考https://zhuanlan.zhihu.com/p/319810661。

### 基本使用

```python
model = MyLightningModule()
trainer = Trainer(max_epochs=1000)
trainer.fit(model, train_dataloader, val_dataloader)
```

在后台，trainer为我们自动完成了以下操作：

- 使用或禁用梯度
- 用loader加载数据
- 调用一些回调函数（保存模型、log等）
- 把数据存在合适的设备上

### Trainer

Trainer在构造时会接受各种参数，可以用构造函数构造也可以用argparse读取。

~~~python
from argparse import ArgumentParser

def main(args):
    model = LightningModule()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)  #开启训练

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
~~~

#### 设备

设置GPU等加速器

~~~ python
# CPU accelerator
trainer = Trainer(accelerator="cpu")

# Training with GPU Accelerator using 2 GPUs
trainer = Trainer(devices=2, accelerator="gpu")

# Training with TPU Accelerator using 8 tpu cores
trainer = Trainer(devices=8, accelerator="tpu")

# Training with GPU Accelerator using the DistributedDataParallel strategy
trainer = Trainer(devices=4, accelerator="gpu", strategy="ddp")
~~~

#### 梯度累积

batch不够大时可以多算几个batch再进行梯度下降

~~~python
# accumulate every 4 batches (effective batch size is batch*4)
trainer = Trainer(accumulate_grad_batches=4)

# no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})
~~~

#### 半精度加速

pl集成了amp和apex利用混合精度加速训练，原理可参考https://blog.csdn.net/HUSTHY/article/details/109485088

~~~python
# using PyTorch built-in AMP, default used by the Trainer
trainer = Trainer(precision=16,amp_backend="native")
# using NVIDIA Apex
trainer = Trainer(amp_backend="apex")
# default used by the Trainer, level可以选择01-03，速度依次加快
trainer = Trainer(amp_level='O2')
~~~

#### 自动寻找合适的batchsize

~~~python
# 默认不开启
trainer = Trainer(auto_scale_batch_size=None)
# 自动找满足内存的 batch size
trainer = Trainer(auto_scale_batch_size=None|'power'|'binsearch')
# 加载到模型
trainer.tune(model)
~~~

#### 设置validate时机

~~~python
# default used by the Trainer
trainer = Trainer(check_val_every_n_epoch=1)
# run val loop every 10 training epochs
trainer = Trainer(check_val_every_n_epoch=10)
~~~

#### CheckPoint

pl会自动保存最近一次迭代的CheckPoint。

~~~~python
#设置默认根目录
trainer = Trainer(default_root_dir="some/path/")
# 是否自动保存
trainer = Trainer(enable_checkpointing=True)
trainer = Trainer(enable_checkpointing=False)
#可以保存所有超参数
class MyLightningModule(LightningModule):
    def __init__(self, learning_rate, another_parameter, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
#导入，可以重写超参数
model=MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt“)
model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)
print(model.learning_rate)

~~~~

#### 梯度裁减

~~~python
# 设置梯度模长最大值，0.0相当于不设置
trainer = Trainer(gradient_clip_val=0.0)
~~~

#### Log （Tensorboard）

~~~python
from pytorch_lightning.loggers import TensorBoardLogger
# default logger used by trainer
logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
Trainer(logger=logger)
~~~

#### Debug相关

~~~python
# 跑训练集的多少比率
trainer = Trainer(limit_train_batches=1.0)
trainer = Trainer(limit_test_batches=0.25)
# run for only 10 batches
trainer = Trainer(limit_val_batches=10)
#是否启用进度条
trainer = Trainer(enable_progress_bar=True)
#多久validate一次，float是epoch的比例（0.25为一个epoch验证4次，int为多少个batch验证一次
# check validation set 4 times during a training epoch
trainer = Trainer(val_check_interval=0.25)
# check validation set every 1000 training batches in the current epoch
trainer = Trainer(val_check_interval=1000)
~~~

### 主要流程函数

~~~python
#训练
Trainer.fit(model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None)
#验证
Trainer.validate(model=None, dataloaders=None, ckpt_path=None, verbose=True, datamodule=None)
#测试
Trainer.test(model=None, dataloaders=None, ckpt_path=None, verbose=True, datamodule=None)
#预测
Trainer.predict(model=None, dataloaders=None, datamodule=None, return_predictions=None, ckpt_path=None)
#在训练前调整超参数，比如设置了自己确定最佳batchsize
Trainer.tune(model, train_dataloaders=None, val_dataloaders=None, datamodule=None, scale_batch_size_kwargs=None, lr_find_kwargs=None)
~~~

### 内置变量

~~~python
#用log将指標保存，可以用callback_metrics读出
def training_step(self, batch, batch_idx):
    self.log("a_val", 2)
callback_metrics = trainer.callback_metrics
assert callback_metrics["a_val"] == 2
#一些变量
if trainer.current_epoch >= 10
if trainer.is_last_batch
#当前路径
def training_step(self, batch, batch_idx):
    img = ...
    save_img(img, self.trainer.log_dir)
~~~

### 复现

~~~python
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)  #第一个是随机数种子，workers如果为True会保证在不同dataloader、numpy、torch中使用不同的随机数。
~~~


