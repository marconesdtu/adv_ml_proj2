import torch
from curve import Curve

def get_all_labels_and_latents(model, data_loader):
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for data, target in data_loader:
              
            latent = model.encoder(data).rsample() 
            all_latents.append(latent)
            all_labels.append(target)

        all_latents = torch.cat(all_latents, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return all_latents, all_labels
                

def plot_curve(ax, curve: Curve, num_steps: int):
    t = torch.linspace(0, 1, steps=num_steps)
    curve_points = curve(t).detach().numpy()

    ax.plot(curve_points[:, 0], curve_points[:, 1], label=curve.__str__(), color='black')