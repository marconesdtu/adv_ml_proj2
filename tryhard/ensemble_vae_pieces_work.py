# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics","ensemble", "ensemble_train"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=1,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net
    

    def energy_function(decoder, curve):

        recon_distribution = decoder(curve)  # Decoder output (likely a distribution)
        recon_means = recon_distribution.mean  # Extract mean values

        diffs = recon_means[:-1] - recon_means[1:]  # Compute successive differences
        energy = torch.sum(diffs ** 2)  # Sum squared differences
        return energy

    def optimize_geodesic(decoder, z1, z2, num_t=20, lr=0.05, steps=500):
        # Initialize curve with small random perturbations around linear interpolation
        t = torch.linspace(0, 1, num_t).view(-1, 1)
        curve = (1 - t) * z1 + t * z2  # Linear interpolation
        curve += 0.1 * torch.randn_like(curve)  # Add small noise to allow flexibility
        curve = curve.clone().detach().requires_grad_(True)  # Enable gradients

        optimizer = optim.Adam([curve], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            # The curve should be passed through the decoder to reconstruct the data
            energy = energy_function(decoder, curve)  # Decode and compute energy
            energy.backward()
            optimizer.step()

            # Optional: Ensure endpoints remain fixed
            with torch.no_grad():
                curve[0] = z1
                curve[-1] = z2

        return curve.detach()


    def plot_geodesics(geodesics):

        plt.figure(figsize=(8, 8))
        
        for curve in geodesics:
            plt.plot(curve[:, 0], curve[:, 1], 'r-', alpha=0.5)  # Geodesic curve
            plt.scatter(curve[:, 0], curve[:, 1], c='blue', s=10)  # Intermediate points

        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title("Optimized Geodesics in Latent Space")
        plt.grid()
        plt.show()


    def train_multiple_vaes(dataloader, num_models=10, epochs=10, save_dir="experiment"):
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(num_models):
            # Create a new VAE model each time
            model = VAE(
                prior=GaussianPrior(M),
                decoder=GaussianDecoder(new_decoder()),
                encoder=GaussianEncoder(new_encoder())
            ).to(device)  # Ensure the model is moved to the correct device

            save_path = os.path.join(save_dir, f"model{i}.pt")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Train the model for the specified number of epochs
            train(
                model,
                optimizer,
                dataloader,
                epochs,
                device
            )
            
            # Save the trained model after training
            torch.save(model.state_dict(), save_path)



    def load_vaes(num_models=10, save_dir="experiment"):
        models = []
        for i in range(num_models):
            # Create the model and load the state dict
            model = VAE(
                prior=GaussianPrior(M),
                decoder=GaussianDecoder(new_decoder()),
                encoder=GaussianEncoder(new_encoder())
            ).to(device)  # Ensure the model is on the correct device
            
            model.load_state_dict(torch.load(os.path.join(save_dir, f"model{i}.pt")))
            model.eval()  # Set the model to evaluation mode
            models.append(model)
        
        return models

    def compute_model_averaged_energy(models, data):
        energies = []
        for model in models:
            # Ensure data is on the same device as the model
            data = data.to(device)  # Move data to the device
            with torch.no_grad():
                recon = model.decoder(model.encoder(data).mean).mean  # Reconstruct the data
                energy = ((recon - data) ** 2).sum()  # Compute energy
            energies.append(energy.item())
        
        return np.mean(energies)  # Return the average energy

    def compute_distances(models, latent_pairs):
        euclidean_dists = []
        geodesic_dists = []
        
        for model in models:
            model = model.to(device)  # Ensure model is on the correct device
            
            for z1, z2 in latent_pairs:
                # Ensure latent vectors are on the same device as the model
                z1, z2 = torch.tensor(z1, dtype=torch.float32).to(device), torch.tensor(z2, dtype=torch.float32).to(device)
                
                # Compute Euclidean distance
                euclidean_dist = torch.norm(z1 - z2).item()
                geodesic_dist = model.compute_geodesic(z1, z2)  # Assuming compute_geodesic method exists
                euclidean_dists.append(euclidean_dist)
                geodesic_dists.append(geodesic_dist)
        
        return euclidean_dists, geodesic_dists

    # Compute Coefficient of Variation (CoV, Eq. 2)
    def compute_cov(distances):
        return np.std(distances) / np.mean(distances)

    # Plot CoV vs number of decoders
    def plot_cov_vs_decoders(cov_values, decoder_counts):
        plt.figure()
        plt.plot(decoder_counts, cov_values, marker='o')
        plt.xlabel("Number of Decoders")
        plt.ylabel("CoV")
        plt.title("CoV vs Number of Decoders")
        plt.show()


    
    # Generate fixed test point pairs with a seed
    def generate_fixed_test_pairs(num_pairs=10, latent_dim=2, seed=42):
        np.random.seed(seed)
        return [(np.random.randn(latent_dim), np.random.randn(latent_dim)) for _ in range(num_pairs)]




    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )


    elif args.mode == "ensemble_train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_multiple_vaes(
            mnist_train_loader, num_models=10, epochs=10, save_dir="experiment"
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        # Generate random latent pairs
        num_pairs = 25  # Number of geodesics to compute
        z_samples = model.prior().sample((num_pairs * 2,)).to(device)
        latent_pairs = [(z_samples[i], z_samples[i + num_pairs]) for i in range(num_pairs)]

        # Compute geodesics
        geodesics = []
        for z1, z2 in latent_pairs:
            curve = optimize_geodesic(model.decoder, z1, z2, num_t=args.num_t)
            geodesics.append(curve.cpu().numpy())

        # Plot the results
        plot_geodesics(geodesics)
        plt.savefig("geodesics.png")

    # elif args.mode == "ensemble":
        # models = load_vaes(num_models=10)
        # test_pairs = generate_fixed_test_pairs()  # Ensure this function generates valid latent pairs

        # # Example input data (latent vector should be passed through the decoder)
        # sample_data = torch.randn(1, M).to(device)  # Latent vector of size [1, M] moved to the correct device
        
        # # Compute model-averaged energy
        # avg_energy = compute_model_averaged_energy(models, sample_data)
        # print(f"Model-Averaged Energy: {avg_energy}")
        
        # # Compute the distances (Euclidean and Geodesic) between test pairs
        # euclidean_dists, geodesic_dists = compute_distances(models, test_pairs)
        
        # # Compute CoVs
        # euclidean_cov = compute_cov(euclidean_dists)
        # geodesic_cov = compute_cov(geodesic_dists)
        
        # # Plot CoV vs number of decoders
        # plot_cov_vs_decoders([euclidean_cov, geodesic_cov], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # print(f"Euclidean CoV: {euclidean_cov}, Geodesic CoV: {geodesic_cov}")





