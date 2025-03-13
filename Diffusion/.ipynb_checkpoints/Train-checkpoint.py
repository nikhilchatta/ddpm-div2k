import torch
import torch.nn as nn
import torch.optim as optim
from Diffusion.Unet import UNet
from Scheduler import GradualWarmupScheduler

def train(modelConfig, train_loader):
    device = modelConfig["device"]
    print(f"Training on device: {device}")

    # ==== Model ====
    net_model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    # ==== Optimizer ====
    optimizer = optim.Adam(net_model.parameters(), lr=modelConfig["lr"])

    # ==== Learning Rate Scheduler ====
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=modelConfig["epoch"])
    warmup_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["warmup_epoch"],
        after_scheduler=cosine_scheduler
    )

    # ==== Loss Function ====
    criterion = nn.MSELoss()

    # ==== Training Loop ====
    net_model.train()
    for epoch in range(modelConfig["epoch"]):
        running_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()

            # ✅ Correct forward pass with timestep
            t = torch.randint(0, modelConfig["T"], (images.size(0),), device=device)
            outputs = net_model(images, t)

            loss = criterion(outputs, images)

            loss.backward()
            nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
            optimizer.step()

            running_loss += loss.item()

        warmup_scheduler.step()
        print(f"Epoch [{epoch + 1}/{modelConfig['epoch']}], Loss: {running_loss:.4f}")

    # ==== Save Model Weights ====
    save_path = modelConfig["save_weight_dir"] + "final_model.pt"
    torch.save(net_model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")
