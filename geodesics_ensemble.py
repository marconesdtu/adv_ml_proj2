import torch
from geodesics import nonlinear_remainder, curve


def energy(weights, decoders, c0, c1, N, num_models):
    """
    Computes the energy approximation of the curve for multiple models using Monte Carlo estimation.
    (eq. 8.7 in the book)
    """
    t = torch.linspace(0, 1, N + 1)
    c_points = curve(t, c0, c1, weights)

    total_energy = 0.0
    
    # Loop over all decoders and compute the energy for each model
    for i in range(num_models):
        decoder_fun = decoders[i]
        f_vals = decoder_fun(c_points)
        diffs = f_vals[1:] - f_vals[:-1]
        total_energy += (diffs ** 2).sum()
    
    # Average the energy over all models (Monte Carlo estimation)
    return total_energy / num_models

def compute_geodesic_ensemble(c0, c1, decoders, num_models, polyn_order=4, latent_dim=2, N=100, num_iterations=2500, lr=1e-3, debug=False):
    """
    Computes an approximate geodesic between two latent points using energy minimization via Adam.
    Includes optional early stopping, and Monte Carlo estimation of the energy for multiple models.
    """
    weights = torch.randn(latent_dim, polyn_order - 1, requires_grad=True)
    
    # Compute the energy for the geodesic using all decoders and average the results
    if debug:
        print(f'Init Energy for {weights}: {energy(weights, decoders, c0, c1, N, num_models).item()}')
    
    optimizer = torch.optim.Adam([weights], lr)
    
    best_energy = float('inf')
    steps_since_improvement = 0

    for i in range(num_iterations):
        optimizer.zero_grad()
        E = energy(weights, decoders, c0, c1, N, num_models)
        E.backward()
        optimizer.step()

        current_energy = E.item()

        # Early stopping based on energy
        if best_energy - current_energy > 1e-4:
            best_energy = current_energy
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if steps_since_improvement >= 200:
            if debug:
                print(f"Early stopping at iteration {i}, Energy: {current_energy}")
            break
        
        if i % 100 == 0 and debug:
            print(f"Iteration {i}, Energy: {current_energy}")
    
    if debug:
        print(f"Final weights: {weights}. Final Energy: {energy(weights, decoders, c0, c1, N, num_models).item()}")
    
    return curve(torch.linspace(0, 1, N + 1), c0, c1, weights)