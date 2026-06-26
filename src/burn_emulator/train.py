import pandas as pd
import time
import torch
import torch.nn as nn

from pathlib import Path
from typing import Any
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from burn_emulator.utils import dynamic_import, save_checkpoint
from burn_emulator.constants import DTYPE, OUTDIR

    
def train_model(
    model: nn.Module,
    num_epochs: int,
    train_loader: DataLoader,
    model_name: str,
    optimizer: Optimizer,
    criterion: nn.Module,
    outpath: Path,
) -> None:
    train_top3 = []
    step = 0
    
    model.to('cuda', dtype=DTYPE)
    model.train()
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        train_loss_avg = 0
        log = []
        for X, Y, M in train_loader:
            X = X.to('cuda', dtype=DTYPE)
            Y = Y.to('cuda', dtype=DTYPE)
            M = M.to('cuda', dtype=DTYPE)

            # no validation loop for now
            # validation is done post training since there is not spatial OOD
            optimizer.zero_grad()
            Y_hat = model(X)
            train_loss = criterion(Y_hat*M, Y*M)
            train_loss.backward()
            optimizer.step()

            step += 1
            train_loss_avg += train_loss.item()

        train_loss_avg /= len(train_loader)
        log.append({"train_loss_avg": train_loss_avg, "epoch": epoch, "step": step, "time": time.perf_counter()-start_time})
        save_checkpoint(model, f"{model_name}_train", epoch, step, train_loss_avg, train_top3, outpath)
        header = False if (outpath / "train_log.csv").exists() else True
        pd.DataFrame(log).to_csv(outpath / "train_log.csv", mode="a", index=False, header=header)
    save_checkpoint(model, f"{model_name}_train", epoch, step, train_loss_avg, [], outpath)


def train(**kwargs: Any) -> None:
    # kwargs are real lazy
    model = dynamic_import(kwargs.get("model"))
    dataset = dynamic_import(kwargs.get("dataset"))
    optimizer = dynamic_import(kwargs.get("optimizer"), {"params": model.parameters()})
    criterion = dynamic_import(kwargs.get("criterion"))
    train_loader = dynamic_import(kwargs.get("dataloader"), {"dataset": dataset})
    
    experiment_path = OUTDIR / kwargs.get("model_name")
    experiment_path.mkdir(exist_ok=True, parents=True)
    
    train_model(model=model,
                num_epochs=kwargs.get("num_epochs"),
                train_loader=train_loader,
                model_name=kwargs.get("model_name"),
                optimizer=optimizer,
                criterion=criterion,
                outpath=experiment_path)