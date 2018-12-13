import numpy as np
from cvxopt import matrix,solvers
import matplotlib.pyplot as plt

def solve_SVM_dual_CVXOPT(x_train, y_train, x_test, C=1):
    (n, d)=x_train.shape
    (m, one)=x_test.shape
    if n==0 or d==0 or m ==0:
        print('Input does not make sense')
        return 0,0
    y_pred_test = np.empty(m);
    K = y_train[:, None] * x_train
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((n, 1)))
    tmp1 = np.diag(np.ones(n) * -1)
    tmp2 = np.identity(n)
    G = matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(n)
    tmp2 = np.ones(n) * C
    h = matrix(np.hstack((tmp1, tmp2)))
    A = matrix(y_train.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x'])
    w = np.sum(alpha * y_train[:, None] * x_train, axis = 0)
    cond = (alpha > 1e-4).reshape(-1)
    tempb = y_train[cond] - np.dot(x_train[cond], w)
    normm = np.linalg.norm(w)
    w, b = w / normm, b / normm
    for i in range(len(x_test)):
        if np.dot(x_test[i,:],w)+b>0:
            y_pred_test[i] = 1
        else:
            y_pred_test[i] = -1
    return y_pred_test, alpha

def solve_SVM_dual_SMO(x_train, y_train, x_test, C=1):
    (n, d)=x_train.shape
    (m, one)=x_test.shape
    if n==0 or d==0 or m ==0:
        print('Input does not make sense')
        return 0,0
    epsilon = 0.001
    alpha = np.zeros((n))
    count = 0
    while True:
        count += 1
        alpha_prev = np.copy(alpha)
        for j in range(0, n):
            i = j
            cnt=0
            while i == j and cnt<1000:
                i = np.random.randint(0,n-1)
                cnt=cnt+1
                
            x_i, x_j, y_i, y_j = x_train[i,:], x_train[j,:], y_train[i], y_train[j]
            k_ij = np.dot(x_i, x_i) + np.dot(x_j, x_j) - 2 * np.dot(x_i, x_j)
            if k_ij == 0:
                continue
            alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
            if(y_i != y_j):
                (L, H) = (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
            else:
                (L, H) = (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

            w = np.dot(x_train.T, np.multiply(alpha,y_train))
            b = np.mean(y_train - np.dot(w.T, x_train.T))


            E_i = np.sign(np.dot(w.T, x_i.T) + b).astype(int) - y_i
            E_j = np.sign(np.dot(w.T, x_j.T) + b).astype(int) - y_j
            alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)

            alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

        diff = np.linalg.norm(alpha - alpha_prev)
        if diff < epsilon:
            break

        if count >= 100000000000:
            break
    w = np.dot(x_train.T, np.multiply(alpha,y_train))
    b = np.mean(y_train - np.dot(w.T, x_train.T))
    normm = np.linalg.norm(w)
    w, b = w / normm, b / normm
    for i in range(len(x_test)):
        if np.dot(x_test[i,:],w)+b>0:
            y_pred_test[i] = 1
        else:
            y_pred_test[i] = -1
    return y_pred_test, alpha