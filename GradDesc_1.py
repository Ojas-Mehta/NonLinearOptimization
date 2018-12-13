import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
tol = 1e-6

def gradient_descent_update(x, eta):
    movement = -1*eta * np.array(get_gradient(x))
    return x + movement

def descent_update_AG(x, desc_dir, alpha=0.25, beta=0.50):
    
    grad = np.array(get_gradient(x))
    val = np.array(get_value(x))
    
    cand_eta = 1
    min_eta = 0 
    max_eta = np.inf
    if np.dot(grad,desc_dir)>=0:
        print('Not a descent direction')
        return x
    while True:
        xnext = x + cand_eta * desc_dir
        f = get_value(xnext)
        f1 = val + beta * cand_eta * np.dot(grad,desc_dir) 
        f2 = val + alpha * cand_eta * np.dot(grad, desc_dir)
        if f>=f1 and f<=f2:
            break
            
        if f<f1: 
            min_eta = cand_eta
            cand_eta = min(float(2*cand_eta), float(cand_eta+max_eta)/2) 
        if f>f2:
            max_eta = cand_eta
            cand_eta = (cand_eta+min_eta)/2 
    return xnext
    
def descent_update_FR(x, desc_dir):
    
    cand_eta = 1.0
    min_eta = 0 
    max_eta = np.inf
    grad = np.array(get_gradient(x))
    x = np.array(x)
    if np.linalg.norm(grad)<1e-7:
        return x
    prev_grad = grad
    if np.dot(grad,desc_dir)>=0:
        print('Not a descent direction')
        return x
    diff_eta = 1
    while min_eta<max_eta and diff_eta>1e-8:
        xnext = np.subtract(x , (cand_eta * (-desc_dir)))
        grad = get_gradient(xnext)
        
        dott = np.dot(grad, prev_grad)
        
        if np.linalg.norm(grad)<1e-7:       
            break
            
        if dott>0:
            min_eta = cand_eta
            old_eta = cand_eta
            cand_eta = min(float(2*cand_eta), float(cand_eta+max_eta)/2) 
        else:
            max_eta = cand_eta
            old_eta = cand_eta
            cand_eta = (cand_eta+min_eta)/2 
        diff_eta = abs(old_eta - cand_eta)
    
    return xnext
    

def BFGS_update(H, s, y):
   
    rho_k = (1/np.dot(np.transpose(s),y))
    M1 = np.dot(rho_k,np.outer(s, y))
    M2 = np.dot(rho_k,np.outer(y, s))
    M3 = np.dot(rho_k,np.outer(s, s))
    dim = M1.shape[0]
    I1 = np.identity(dim)
    I2 = np.identity(dim)
    
    Hnew = np.add(np.dot((np.dot((I1 - M1),H)), I2-M2), M3)
    return Hnew



def gradient_descent(x0, num_iter=100, eta='AG'):
 
    if eta == 'FR':
        for i in range(num_iter):
            grd = get_gradient(x0)
            x0 = descent_update_FR(x0, -grd)
        return x0
    
    elif eta == 'AG':
        for i in range(num_iter):
            grd = get_gradient(x0)
            x0 = descent_update_AG(x0, -grd)
        return x0
    else:
        for i in range(num_iter):
            x0 = gradient_descent_update(x0, eta)
        return x0

def quasi_Newton(x0, H0, num_iter=100, eta='AG'):
   
   # x1 = np.array([0.,0.,0.])
    if eta == 'FR':
        for i in range(num_iter):
            grad = get_gradient(x0)
        
            x1 = descent_update_FR(x0, -np.dot(H0,grad))
            sk = x1 - x0
            f1 = get_gradient(x1)
            f0 = grad
            yk = f1 - f0
            if(np.linalg.norm(sk)<1e-7):
                break
            H1 = BFGS_update(H0, sk, yk)
            H0 = H1
            x0 = x1
        return x1
    
    elif eta == 'AG':
        for i in range(num_iter):
            grad = get_gradient(x0)
        
            x1 = descent_update_AG(x0, -np.dot(H0,grad))
            sk = x1 - x0
            f1 = get_gradient(x1)
            f0 = get_gradient(x0)
            yk = f1 - f0
            if(np.linalg.norm(sk)<1e-7):
                break
            H1 = BFGS_update(H0, sk, yk)
            H0 = H1
            x0 = x1
        return x1
    else:
        for i in range(num_iter):
            grad = get_gradient(x0)
            
            x1 = x0 - 1*eta * np.array(np.dot(H0,grad))
            sk = x1 - x0
            f1 = get_gradient(x1)
            f0 = grad
            yk = f1 - f0
            if(np.linalg.norm(sk)<1e-7):
                break
            H1 = BFGS_update(H0, sk, yk)
            H0 = H1
            x0 = x1
        return x1
    


    
