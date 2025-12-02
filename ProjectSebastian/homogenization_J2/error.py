from scipy.sparse import load_npz
import numpy as np

# Load data
K_vec  = load_npz("K_ff_vectorized.npz")
K_ref  = load_npz("K_ff_nonvectorized.npz")

R_vec  = np.load("R_f_vectorized.npy")
R_ref  = np.load("R_f_nonvectorized.npy")

du_vec = np.load("delta_u_f_vectorized.npy")
du_ref = np.load("delta_u_f_nonvectorized.npy")

diff_K = K_vec - K_ref
print("||K diff||_inf =", abs(diff_K).max())
print("nnz(K diff)    =", diff_K.nnz)

print("||R_f diff||_inf =", np.max(np.abs(R_vec - R_ref)))

print("||delta_u_f diff||_inf =", np.max(np.abs(du_vec - du_ref)))

import numpy as np

sigma_non = np.load("sigma_nonvec_elem1.npy")   # (n_gp, 3)
Ctan_non  = np.load("Ctan_nonvec_elem1.npy")    # (n_gp, 3, 3)

sigma_vec = np.load("sigma_vec_elem1.npy")      # (n_gp, 3)
Ctan_vec  = np.load("Ctan_vec_elem1.npy")       # (n_gp, 3, 3)

print("sigma max diff:", np.max(np.abs(sigma_vec - sigma_non)))
print("Ctan  max diff:", np.max(np.abs(Ctan_vec  - Ctan_non)))

# you can also print one Gauss point
g0 = 0
print("sigma_non[g0]:\n", sigma_non[g0])
print("sigma_vec[g0]:\n", sigma_vec[g0])
print("Ctan_non[g0]:\n", Ctan_non[g0])
print("Ctan_vec[g0]:\n", Ctan_vec[g0])


