import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .generator import UNetGenerator
from .discriminator import Discriminator
from .dataset import PairedFloorPlanDataset

def train_symbol_placement_gan(empty_dir, symbols_dir, num_epochs=100, batch_size=4, save_dir="output_gan", lambda_l1=30):
    dataset = PairedFloorPlanDataset(empty_dir, symbols_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = UNetGenerator().to(device)
    D = Discriminator().to(device)
    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    g_optim = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.999))
    d_optim = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        G.train()
        total_g_loss, total_d_loss = 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            fake = G(inputs)

            # Discriminator
            D.zero_grad()
            real_preds = D(inputs, targets)
            fake_preds = D(inputs, fake.detach())
            d_loss = (criterion_gan(real_preds, torch.ones_like(real_preds)) +
                      criterion_gan(fake_preds, torch.zeros_like(fake_preds))) * 0.5
            d_loss.backward()
            d_optim.step()

            # Generator
            G.zero_grad()
            fake_preds = D(inputs, fake)
            g_loss = criterion_gan(fake_preds, torch.ones_like(fake_preds))
            l1_loss = criterion_l1(fake, targets) * lambda_l1
            total_loss = g_loss + l1_loss
            total_loss.backward()
            g_optim.step()

            total_g_loss += total_loss.item()
            total_d_loss += d_loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - G_Loss: {total_g_loss:.4f}, D_Loss: {total_d_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), os.path.join(save_dir, f"G_epoch_{epoch+1}.pth"))