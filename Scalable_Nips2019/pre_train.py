import argparse
import os
import yaml
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler, Adam, SGD

from TCN import TCN
from Triplet_Loss import TripletLoss
from Dataset import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_unsupervised(
    encoder: torch.nn.Module,
    dataloader: DataLoader,
    criterion: TripletLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    neg_pool_tensor: torch.Tensor,
    save_dir: str = None,
    save_epoch: int = 20,
    grad_clip: float = None,
    scheduler=None,
):
    encoder.train()
    neg_pool_tensor = neg_pool_tensor.to(device, non_blocking=True)
    best_loss = float('inf')  

    if save_dir:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)


    epoch_pbar = tqdm(range(epochs), desc="Training Progress", position=None, leave=True, ncols=None)
    
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        num_batches = 0
        ema = None  

        batch_pbar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1:3d}/{epochs}", 
            position=None, 
            leave=True, 
            unit="batch",
            ncols=None, 
            miniters=1 
        )
        
        for batch_data in batch_pbar:

            x = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            loss = criterion(x, encoder, neg_pool_tensor)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

            optimizer.step()

            val = float(loss.item())
            ema = val if ema is None else 0.9 * ema + 0.1 * val
            epoch_loss += val

            num_batches += 1
            
            batch_pbar.set_postfix({
                'loss': f"{val:.4f}",
                'ema': f"{ema:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(1, num_batches)
        
        epoch_pbar.set_postfix({
            'avg_loss': f"{avg_loss:.6f}",
            'final_ema': f"{ema:.6f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        tqdm.write(f"Epoch {epoch+1:3d} Summary: avg_loss={avg_loss:.6f}, ema={ema:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'ema': ema,
                'lr': optimizer.param_groups[0]['lr']
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            best_path = os.path.join(exp_dir, "best.pth")
            torch.save(checkpoint, best_path)
            tqdm.write(f"已保存最佳模型: best.pth (loss: {avg_loss:.6f})")

        if save_dir and ((epoch + 1) % save_epoch == 0 or epoch + 1 == epochs):
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'ema': ema,
                'lr': optimizer.param_groups[0]['lr']
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            latest_path = os.path.join(exp_dir, "latest.pth")
            torch.save(checkpoint, latest_path)
            
            if epoch + 1 == epochs:
                final_path = os.path.join(exp_dir, "final.pth")
                torch.save(checkpoint, final_path)
                tqdm.write(f"🎯 训练完成，已保存最终模型: final.pth")
        
    epoch_pbar.close()
    tqdm.write("Training completed!")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--encoder", type=str, default="TCN")
    parser.add_argument("--dataset", type=str, default="ucr")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load dataset
    dataset_name = args.dataset
    data_config = config["data"]["datasets"].get(dataset_name, {})
    data_path = data_config.get("path")
    
    # Get dataset based on args and config
    dataset = get_dataset(
        dataset_name="mock",
        data_path=data_path,
    )
    dataloader = DataLoader(dataset, **config["data"]["common"])

    # Initialize model with dynamic in_channels
    model_config = config["model"]["kwargs"].copy()
    model = TCN(**model_config).to(device)

    # Create TripletLoss
    criterion = TripletLoss(**config["loss"]["kwargs"])

    # Create optimizer
    optim_name = config["optim"]["name"]
    optim_kwargs = config["optim"]["kwargs"]
    if optim_name.lower() == "adam":
        optimizer = Adam(model.parameters(), **optim_kwargs)
    elif optim_name.lower() == "sgd":
        optimizer = SGD(model.parameters(), **optim_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    # Setup gradient clipping if specified
    grad_clip = config["optim"].get("grad_clip", None)
    
    # Setup scheduler if specified
    scheduler = None
    scheduler_name = config["optim"].get("scheduler", None)
    if scheduler_name:
        scheduler_kwargs = config["optim"].get("scheduler_kwargs", {})
        scheduler = getattr(lr_scheduler, scheduler_name)(optimizer, **scheduler_kwargs)

    # Prepare negative pool tensor (entire dataset for negative sampling)
    all_data = []
    for batch in dataloader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        all_data.append(x)
    neg_pool_tensor = torch.cat(all_data, dim=0)
    
    # 获取trainer配置
    trainer_config = config["trainer"]
    epochs = trainer_config.get("epochs", 100)
    save_dir = trainer_config.get("save_dir", None)
    save_epoch = trainer_config.get("save_epoch", 20)
    
    # Train model
    train_unsupervised(
        encoder=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        neg_pool_tensor=neg_pool_tensor,
        save_dir=save_dir,
        save_epoch=save_epoch,
        grad_clip=grad_clip,
        scheduler=scheduler
    )


if __name__ == "__main__":
    main()

