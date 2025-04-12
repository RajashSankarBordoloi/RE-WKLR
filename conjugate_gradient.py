
import numpy as np


def conjugate_gradient(A, b, x0, max_iter=200, tol=0.005):
    
    x = x0.copy()
    r = b - A @ x
    d = r.copy()
    c = 0
    
    while np.linalg.norm(r)**2 > tol and c < max_iter:
        if c == 0:
            zeta = 0
        else:
            zeta = np.dot(r, r) / np.dot(r, r_prev)
            
        d = r + zeta * d
        Ad = A @ d
        r_prev = r.copy()
        s = np.dot(r_prev, r_prev) / np.dot(d, Ad)
        
        x = x + zeta * d
        r = r_prev - s * Ad
        
        c += 1
        
        return x

