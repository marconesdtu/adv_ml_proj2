import torch

def nonlinear_remainder(t, weights):
    """
    Computes the nonlinear part (c tilde) of the geodesic curve given time steps, weights, and polynomial order.
    (eq. 8.3 in the book)
    """
    polyn_order = weights.shape[1] + 1 # K in the book

    t_powers = torch.stack([t ** (k + 1) for k in range(polyn_order)], dim=0)
    
    c_tilde_free = torch.matmul(weights, t_powers[:polyn_order - 1])
    
    c_tilde_last = - torch.sum(weights, dim=1, keepdim=True) * t_powers[-1]
    
    return c_tilde_free + c_tilde_last

def curve(t, c0, c1, weights):
    """
    Constructs a curve between two latent points
    (eq. 8.2 in the book)

    """
    linear_part = (1 - t).unsqueeze(1) * c0 + t.unsqueeze(1) * c1
    
    return linear_part + nonlinear_remainder(t, weights).T

def energy(weights, function, c0, c1, N):
    """
    Computes the energy approximation of the curve.
    (eq. 8.7 in the book)
    """
    t = torch.linspace(0, 1, N + 1)
    c_points = curve(t, c0, c1, weights)
    
    f_vals = function(c_points) 
    diffs = f_vals[1:] - f_vals[:-1]
    return (diffs ** 2).sum()

def compute_geodesic(
    c0, 
    c1, 
    decoder_fun, 
    polyn_order=4, 
    latent_dim=2, N=100, 
    num_iterations=2500, 
    lr=1e-3, 
    debug=False,
    early_stopping=True, 
    patience=200, 
    tolerance=1e-4
):
    """
    Computes an approximate geodesic between two latent points using energy minimization via Adam.
    Includes optional early stopping.
    """
    weights = torch.randn(latent_dim, polyn_order - 1, requires_grad=True)
    
    if debug:
        print(f'Init Energy for {weights}: {energy(weights, decoder_fun, c0, c1, N).item()}')
    
    optimizer = torch.optim.Adam([weights], lr)
    
    best_energy = float('inf')
    steps_since_improvement = 0

    for i in range(num_iterations):
        optimizer.zero_grad()
        E = energy(weights, decoder_fun, c0, c1, N)
        E.backward()
        optimizer.step()

        current_energy = E.item()

        if early_stopping:
            if best_energy - current_energy > tolerance:
                best_energy = current_energy
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if steps_since_improvement >= patience:
                if debug:
                    print(f"Early stopping at iteration {i}, Energy: {current_energy}")
                break
        
        if i % 100 == 0 and debug:
            print(f"Iteration {i}, Energy: {current_energy}")
    
    if debug:
        print(f"Final weights: {weights}. Final Energy: {energy(weights, decoder_fun, c0, c1, N).item()}")
    
    return curve(torch.linspace(0, 1, N + 1), c0, c1, weights)