import torch
from curve import Curve

def energy(curve: Curve, decoder_fun, device, num_steps=100):
    """
    Computes the energy approximation of the curve.
    (eq. 8.7 in the book)
    """
    t = torch.linspace(0, 1, num_steps + 1, device=device)
            
    f_vals = decoder_fun(curve(t))
    
    diffs = f_vals[1:] - f_vals[:-1] 
    
    energy = (diffs ** 2).sum(dim=-1).sum()
    
    return energy


def compute_geodesic(c0, c1, decoder_fun, device, polyn_order=3, latent_dim=2, N=100, num_iterations=2000, lr=1e-3, debug=False):
    """
    Computes an approximate geodesic between two latent points using energy minimization via Adam.
    (eq. 8.10 in the book)
    """
    curve = Curve(c0=c0, c1=c1, polyn_order=polyn_order).to(device)
    
    optimizer = torch.optim.Adam(curve.parameters(), lr=lr, eps=1e-8)
        
    for i in range(num_iterations):
        optimizer.zero_grad()
        E = energy(curve, decoder_fun, device)
        E.backward()
        optimizer.step()
        if i % 100 == 0 and debug:
           print(f"Iteration {i}, Energy: {E.item()}")
    
    return curve
