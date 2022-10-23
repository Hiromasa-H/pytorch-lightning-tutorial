# Pytorch Lightning Tutorial

Pytorch Lightning について学ぶためのレポ。

## インストール
```
pip install pytorch-lightning
```

## 使い方
参考：[https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html)

以下のコードは全て[basic_usage.py](basic_usage.py)にまとめてある。

### 1. LightningModuleの定義
LightningModuleを使うことで、training_step内で`nn.Module`を操作することができる。
以下のサンプルコードではencoder層とdecoder層をそれぞれ`nn.Sequential()`定義し、それらを引数として受け取る`LitAutoEncoder()`内でそれらの学習方法（`training_step()`）を定義している。

```python
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
```

### 2. Datasetの定義
```python
#setup data
dataset = MNIST(os.getcwd(),download=True,transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)
```

### 3. モデルの訓練
`Trainer` を用いるととで簡単に学習ができる
```python
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder,train_dataloaders=train_loader)
```

### 4. モデルを使った推論

```python
# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = Tensor(4, 28 * 28)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
```

### 5. Tensorboardを使った学習の可視化
```bash
$ tensorboard --logdir .
```

### 6. 学習方法の指定
```python
# train on 4 GPUs
trainer = Trainer(
    devices=4,
    accelerator="gpu",
 )

# train 1TB+ parameter models with Deepspeed/fsdp
trainer = Trainer(
    devices=4,
    accelerator="gpu",
    strategy="deepspeed_stage_2",
    precision=16
 )

# 20+ helpful flags for rapid idea iteration
trainer = Trainer(
    max_epochs=10,
    min_epochs=5,
    overfit_batches=1
 )

# access the latest state of the art techniques
trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])
```
- ただし現在GPUに搭載しているCUDAバージョンの関係で、DeepSpeedは上の通りには使え無さそう。
