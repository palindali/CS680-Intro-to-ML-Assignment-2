

import numpy as np
import pandas as pd
from scipy import stats
from sympy import Q
from tqdm import tqdm
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------
########## Q3.3 Sequential Adaboost Implementation
# Input:
# M: matrix (nxd) where elements are (yi*hj(xi))
# w: vector (dx1) initial w values
# p: vector (nx1) initial w values
# max_pass: integer: the number of passes
# Output:
# w: vector of size d, representing the final values of w after training 
def seqadaboost(M, M_tst, max_pass=300):
    n, d = M.shape

    # Initialize w and p
    w = np.zeros(d)
    p = np.ones(n)
    
    # Error metrics
    trn_loss = np.zeros(max_pass)
    trn_error = np.zeros(max_pass)
    tst_error = np.zeros(max_pass)

    # Run algorithm
    M_n = M
    # Preprocess M (Normalize)
    M_n = M_n / (d)
    neg_part = np.vectorize(lambda x : -x if (x < 0) else 0)
    pos_part = np.vectorize(lambda x : x if (x > 0) else 0)
    for t in tqdm(range(max_pass), "Passes", leave=False):
    # for t in range(max_pass):
        p = p / p.sum()
        # epsil = np.matmul(neg_part(M_n).T, p)
        gamma = np.matmul(pos_part(M_n).T, p)
        epsil = np.matmul(abs(M_n).T, p) - gamma
        beta = 0.5 * (np.log(gamma) - np.log(epsil))

        # Choose classifier
        jt = np.argmax(abs(np.sqrt(epsil) - np.sqrt(gamma)))
        alpha = np.zeros(d)
        alpha[jt] = 1

        w = w + alpha*beta
        p = p * np.exp(np.matmul(-M_n, alpha*beta))
        
        # Calculate metrics
        # Training Loss
        M_nw = np.matmul(M_n, w)
        trn_loss[t] = np.exp(-M_nw).sum()
        # Training Error
        Mw = np.matmul(M, w)
        trn_error[t] = (Mw <= 0).sum() / n
        # Testing Error
        Mw_tst = np.matmul(M_tst, w)
        tst_error[t] = (Mw_tst <= 0).sum() / M_tst.shape[0]
        
        print(f"Pass {t}:")
        print(f"Training Loss: {trn_loss[t]}")
        print(f"Training Error: {trn_error[t]}")
        print(f"Testing Error: {tst_error[t]}")
        print(f"")
    
    # Output plot
    it = list(range(max_pass))
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    lns1 = ax1.plot(it, trn_loss, 'r-', label = "Training Loss")
    lns2 = ax2.plot(it, trn_error, 'b-', label = "Training Error")
    lns3 = ax2.plot(it, tst_error, 'g-', label = "Testing Error")

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Error')
    
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
    plt.title("Q3.4 Sequential Adaboost")
    plt.show()

    return w


M = np.load('M.npy')
M_tst = np.load('M_tst.npy')
seqadaboost(M, M_tst)
