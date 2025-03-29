import torch

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
                