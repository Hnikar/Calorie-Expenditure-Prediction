import os
import logging
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .workout_dataset import WorkoutDataset
from .model import NeuralNetwork
from .rmsle_loss import RMSLELoss

logger: logging.Logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    momentum: float = 0.9
) -> None:
    
    criterion: nn.Module = RMSLELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_rmsle: float = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss: float = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if epoch % 5 == 0:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    pred = model(X_val)
                    val_loss = criterion(pred, y_val)
                    val_loss_sum += val_loss.item()
            
            avg_val_rmsle = val_loss_sum / len(val_loader)
            val_losses.append(avg_val_rmsle)

            logger.info(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val RMSLE: {avg_val_rmsle:.4f}")

            if avg_val_rmsle < best_rmsle:
                best_rmsle = avg_val_rmsle
                save_path: str = os.path.join(HydraConfig.get().run.dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Training step")
    plt.ylabel("Loss (RMSLE)")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(HydraConfig.get().run.dir, "loss_plot.png"))
    plt.close()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    set_seed(42)

    full_dataset: Dataset = WorkoutDataset(cfg["data"]["train"])
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)


    sample_x, _ = full_dataset[0]
    input_dim = sample_x.shape[0]

    model: nn.Module = NeuralNetwork(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout_rate=cfg["model"]["dropout"]
    )

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"]
    )

if __name__ == "__main__":
    main()