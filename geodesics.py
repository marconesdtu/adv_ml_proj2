import torch

def nonlinear_remainder(t, weights):
    """
    Computes the nonlinear part (c tilde) of the geodesic curve given time steps, weights, and polynomial order.
    (eq. 8.3 in the book)

    Parameters:
        t (Tensor): 1D tensor of time values (values between 0 and 1)
        weights (Tensor): Weight parameters for the nonlinear term with shape (latent_dim, polyn_order - 1)

    Returns:
        Tensor: Nonlinear remainder with shape (latent_dim, len(t))
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

    Parameters:
        t (Tensor): 1D tensor of time values (values between 0 and 1)
        c0 (Tensor): Starting point in latent space (shape: (latent_dim,))
        c1 (Tensor): Ending point in latent space (shape: (latent_dim,))
        weights (Tensor): Weight parameters for the nonlinear term (shape: (latent_dim, polyn_order - 1))

    Returns:
        Tensor: Curve points of shape (len(t), latent_dim)
    """
    linear_part = (1 - t).unsqueeze(1) * c0 + t.unsqueeze(1) * c1
    
    return linear_part + nonlinear_remainder(t, weights).T

def energy(weights, function, c0, c1, N):
    """
    Computes the energy approximation of the curve.
    (eq. 8.7 in the book)

    Parameters:
        weights (Tensor): Weight parameters for the nonlinear term
        function (callable): A function applied to each point on the curve
        c0 (Tensor): Starting point in latent space
        c1 (Tensor): Ending point in latent space
        N (int): Number of discretized steps in the curve

    Returns:
        Tensor: Scalar energy value of the curve.
    """
    t = torch.linspace(0, 1, N + 1)
    c_points = curve(t, c0, c1, weights)
    
    fun_values = [function(c_points[i]) for i in range(c_points.shape[0])]
    
    energy = torch.tensor(0.0, device=weights.device)
    for s in range(1, len(fun_values)):
        energy += torch.sum((fun_values[s] - fun_values[s - 1]) ** 2)
    return energy

def compute_geodesic(c0, c1, function, polyn_order=4, latent_dim=2, N=100, num_iterations=1000, lr=1e-3, debug=False):
    """
    Computes an approximate geodesic between two latent points using energy minimization via Adam.
    (eq. 8.10 in the book)

    Parameters:
        c0 (Tensor): Starting point in latent space (shape: (latent_dim,))
        c1 (Tensor): Ending point in latent space (shape: (latent_dim,))
        function (callable): A function applied to each point on the curve
        polyn_order (int): Order of the polynomial used to parameterize the curve
        latent_dim (int): Dimensionality of the latent space
        N (int): Number of steps in the discretized curve
        num_iterations (int): Number of Adam optimization iterations
        lr (float): Learning rate for Adam
        debug (bool): Prints additional info if it's True

    Returns:
        Tensor or None: Geodesic curve points of shape (N+1, latent_dim) if optimization is successful
    """
    weights = torch.randn(latent_dim, polyn_order - 1, requires_grad=True)
    
    if debug:
        print(f'Init Energy for {weights}: {energy(weights, function, c0, c1, N).item()}')
    
    optimizer = torch.optim.Adam([weights], lr=lr)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        E = energy(weights, function, c0, c1, N)
        E.backward()
        optimizer.step()
        
        if i % 100 == 0 and debug:
            print(f"Iteration {i}, Energy: {E.item()}")
    
    if debug:
        print(f"Found weights {weights}. Min Energy: {energy(weights, function, c0, c1, N).item()}")
    
    return curve(torch.linspace(0, 1, N + 1), c0, c1, weights)