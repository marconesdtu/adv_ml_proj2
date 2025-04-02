import torch
import numpy as np
from euclidean import euclidean_distance

# Function to compute CoV for Euclidean and Geodesic distances
def calculate_CoV(experiment_folder, model_range, test_pairs, M, device, num_decoders_list=[1, 2, 3], num_iterations=2500, N=100, lr=1e-3):
    """
    Calculate the Coefficient of Variation (CoV) of distances (Euclidean and Geodesic) across multiple models
    with varying numbers of decoders in the ensemble.
    
    Parameters:
    - experiment_folder: Path to the folder containing the trained models.
    - model_range: Range of model indices (e.g., range(0, 10) for model0.pt to model9.pt).
    - test_pairs: List of tuples containing fixed pairs of latent points [(yi, yj)].
    - M: Number of models to evaluate.
    - device: Device (CPU or CUDA) to perform computations on.
    - num_decoders_list: List of decoder counts (e.g., [1, 2, 3]).
    - num_iterations: Number of iterations for energy minimization.
    - N: Number of time steps for the geodesic.
    - lr: Learning rate for optimization.
    
    Returns:
    - CoVs: Dictionary with CoVs for Euclidean and Geodesic distances for each number of decoders.
    """
    # Initialize dictionary to store CoVs for different decoder counts
    CoVs = {1: {'euclidean': [], 'geodesic': []},
            2: {'euclidean': [], 'geodesic': []},
            3: {'euclidean': [], 'geodesic': []}}

    # Function to load models and get the decoders
    def load_models_and_decoders(num_decoders):
        decoders = []
        for i in model_range:
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder())
            ).to(device)
            
            model.load_state_dict(torch.load(f"{experiment_folder}/model{i}.pt", weights_only=True))
            model.eval()
            
            # Append the decoder function for this model
            decoder_fun = lambda x: model.decoder(x).mean
            decoders.append(decoder_fun)
        
        return decoders[:num_decoders]  # Return only the decoders up to the specified number

    # Iterate over different numbers of decoders (1, 2, or 3)
    for num_decoders in num_decoders_list:
        # Load the appropriate decoders for the current ensemble size
        decoders = load_models_and_decoders(num_decoders)

        # Iterate over each test pair
        for pair in test_pairs:
            c0, c1 = pair

            # Calculate the Euclidean distance for each model in the ensemble
            euclidean_distances = []
            for decoder_fun in decoders:
                latent_0 = decoder_fun(c0)
                latent_1 = decoder_fun(c1)
                euclidean_distances.append(euclidean_distance(latent_0, latent_1).item())

            # Calculate the Geodesic distance for each model in the ensemble
            geodesic_distances = []
            for decoder_fun in decoders:
                geodesic = compute_geodesic_ensemble(c0, c1, decoders, num_decoders, num_iterations=num_iterations, N=N, lr=lr)
                geodesic_distances.append(geodesic)

            # Calculate CoV for Euclidean distance
            euclidean_mean = np.mean(euclidean_distances)
            euclidean_std = np.std(euclidean_distances)
            euclidean_cov = euclidean_std / euclidean_mean if euclidean_mean != 0 else 0

            # Calculate CoV for Geodesic distance
            geodesic_mean = np.mean(geodesic_distances)
            geodesic_std = np.std(geodesic_distances)
            geodesic_cov = geodesic_std / geodesic_mean if geodesic_mean != 0 else 0

            # Store CoVs
            CoVs[num_decoders]['euclidean'].append(euclidean_cov)
            CoVs[num_decoders]['geodesic'].append(geodesic_cov)

    return CoVs


# Example usage
experiment_folder = 'path_to_experiment_folder'  # replace with the actual path
model_range = range(0, 10)  # Models from model0.pt to model9.pt
test_pairs = [(torch.randn(2), torch.randn(2)) for _ in range(5)]  # Replace with actual test latent pairs
M = 2  # Latent dimension (example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate CoV for different numbers of decoders
CoVs = calculate_CoV(experiment_folder, model_range, test_pairs, M, device)

# Print the CoVs for review
print(CoVs)