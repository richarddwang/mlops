import random, math
from typing import Any, Optional
import torch
from time import sleep
from math import log2, log, exp
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from peft import get_peft_model, LoraConfig, TaskType
import mlflow

mlflow.pytorch.autolog

peft_config = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=".*1",
)


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(1, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.gelu(x)
        x = self.l2(x)
        x = torch.nn.functional.gelu(x)
        x = self.l3(x)
        return x



class FakeLitModel(L.LightningModule):
    def __init__(self, hidden_size, lr , weight_decay, kk=3):
        super().__init__()
        self.save_hyperparameters()
        self.model = FeedForwardNetwork(hidden_size)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch["x"].unsqueeze(-1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(y_hat, batch["y"])
        self.log("training_loss", loss, prog_bar=True)
        sleep(0.01)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["x"].unsqueeze(-1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(y_hat, batch["y"])
        self.log("validation_loss", loss, on_epoch=True, on_step=False)
        sleep(0.01)

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = torch.tensor(random.randint(0, 100000)).float()
        y = x.sqrt()
        return {"x": x, "y": y}


class FakeLitData(L.LightningDataModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def setup(self, stage: str) -> None:
        self.dataset = FakeDataset(self.size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=False,
        )


litmodel = FakeLitModel(45, 0.001, 0.1)
litdata = FakeLitData(1000)

logger = MLFlowLogger(
        experiment_name="MLops Test",
        run_name="sqrt",
        tracking_uri="http://127.0.0.1:5000",
        log_model=True,
        prefix = "prefix",
)

trainer = L.Trainer(
    logger=logger,
    gradient_clip_val=0.1,
    max_epochs=3,
)

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.pytorch.autolog(log_every_n_step=50)
# with mlflow.start_run(experiment_id='762448250173551023'):
#     trainer.fit(litmodel, litdata)

trainer.fit(litmodel, litdata)