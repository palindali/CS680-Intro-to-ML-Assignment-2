#Q3.2 Code

import pickle
import numpy as np

with open("train_test_split.pkl", "br") as fh:
    data = pickle.load(fh)
train_data = data[0]
test_data = data[1]

train_x = train_data[:,:23]
train_y = train_data[:,23] # labels are either 0 or 1
test_x = test_data[:,:23]
test_y = test_data[:,23] # labels are either 0 or 1

# Transform y's to -1, 1
train_y = train_y*2 - 1
test_y = test_y*2 - 1

n = train_x.shape[0]
d = train_x.shape[1]

s = np.zeros(d)
b = np.zeros(d)
loss = np.zeros(d)
test_error = np.zeros(d)
M = np.zeros((n, d))
M_tst = np.zeros((test_x.shape[0], d))

p = np.full(n, 1/n)
for j in range(d):
    xj = train_x[:, j].copy()

    # Sort vectors to iterate from lowest to highest
    sorting = xj.argsort()
    xj = xj[sorting[::]]
    y = train_y[sorting[::]]
    pj = p[sorting[::]]
    
    # Calculate the incremental loss per point when 
    #  having the threshold cross it
    loss_point = y * pj
    
    # Initialize loss to infinity
    s_vec = []
    b_vec = []
    L_vec = []
    
    # Start with lowest threshold
    threshold = xj[0] - 1
    
    bi = -threshold
    si = 1
    Li = ((y==-1)*pj).sum()
    L_vec.append(Li)
    s_vec.append(si)
    b_vec.append(bi)

    L_vec.append(1 - Li)
    s_vec.append(-si)
    b_vec.append(-bi)

    i=0
    while i < n:
        while True:
            Li += loss_point[i]
            i+=1
            if i >= n or xj[i] != xj[i-1]:
                break
        
        if(i < n):
            threshold = (xj[i-1] + xj[i])/2
        else:
            threshold = xj[i-1] + 1

        bi = -threshold
        si = 1
        L_vec.append(Li)
        s_vec.append(si)
        b_vec.append(bi)

        L_vec.append(1 - Li)
        s_vec.append(-si)
        b_vec.append(-bi)
    
    L_vec = np.asarray(L_vec)
    s_vec = np.asarray(s_vec)
    b_vec = np.asarray(b_vec)
    sorting = L_vec.argsort()
    L_vec = L_vec[sorting[::]]
    s_vec = s_vec[sorting[::]]
    b_vec = b_vec[sorting[::]]

    # m = np.asarray([L_vec, s_vec, b_vec])

    L_best = np.median(L_vec[np.where(L_vec == L_vec[0])])
    s_best = np.median(s_vec[np.where(L_vec == L_vec[0])])
    b_best = np.median(b_vec[np.where(L_vec == L_vec[0])])
    loss[j] = L_best
    s[j] = s_best
    b[j] = b_best

    # Test error
    test_n = test_x.shape[0]
    test_error[j] = (np.full(test_n, 1/test_n) * ((test_y * (s[j]*test_x[:,j] + b[j])) <= 0)).sum()
    sign = np.vectorize(lambda v : 1 if v >=0 else -1)
    M[:, j] = train_y * sign(s[j]*train_x[:,j] + b[j])
    M_tst[:, j] = test_y * sign(s[j]*test_x[:,j] + b[j])

# Best feature
best_ind = np.where(test_error == test_error.min())[0]
test_error_best = test_error[best_ind]
Lj_best = loss[best_ind]
sj_best = s[best_ind]
bj_best = b[best_ind]

print(f"Best Feature: {best_ind + 1}")
print(f"Test Error: {test_error_best}")
print(f"Training Error: {Lj_best}")
print(f"sj: {sj_best}")
print(f"bj: {bj_best}")

# Worst feature
worst_ind = np.where(test_error == test_error.max())[0]
test_error_worst = test_error[worst_ind]
Lj_worst = loss[worst_ind]
sj_worst = s[worst_ind]
bj_worst = b[worst_ind]

print(f"Worst Feature: {worst_ind+1}")
print(f"Test Error: {test_error_worst}")
print(f"Training Error: {Lj_worst}")
print(f"sj: {sj_worst}")
print(f"bj: {bj_worst}")

# Write Ms
np.save('M.npy', M)
np.save('M_tst.npy', M_tst)
