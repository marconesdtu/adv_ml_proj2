import torch
import torch.nn as nn
import uuid


class Curve(nn.Module):
    """
    Represents a curve as per eq 8.2 from the book
    """
    def __init__(self, c0, c1, polyn_order):
        super().__init__()
           
        self.c0 = c0.detach().clone()
        self.c1 = c1.detach().clone()
        
        assert polyn_order >=1, "The order of the polynomial must be >=1"
        self.polyn_order = polyn_order
       
        self.weights = nn.Parameter(torch.randn(polyn_order - 1, c0.shape[0], requires_grad=True))
        

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
    
        last_coeff = -self.weights.sum(dim=0, keepdim=True)  
        coeffs = torch.cat([self.weights, last_coeff], dim=0)

        t_power = torch.stack([t ** (i + 1) for i in range(self.polyn_order)], dim=1)

        c_tilde = (t_power * coeffs.unsqueeze(0)).sum(dim=1)

        return (1 - t) * self.c0 + t * self.c1 + c_tilde
    
    def __str__(self):
        return f'Curve: {self.c0.detach().numpy()}, {self.c1.detach().numpy()}'
    
    
