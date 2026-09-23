import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from importlib import reload

from HoneycombSU2.HC_1_hole import StringBasisHC as basis_hc_1h
from HoneycombSU2.HC_2_holes import StringBasis as basis_hc_2h
from HoneycombSU2 import helper_sc_cc_overlaps
reload(helper_sc_cc_overlaps)
from HoneycombSU2.helper_sc_cc_overlaps import make_triangular_grid_bz

#set system parameters
l_sc = 5
initial_sl = 0
l_cc = 5
n_bands = 1

t = 1
J = 0.3
J_perp = 0.3

#define Krylov basis
basis_1 = basis_hc_1h(depth=l_sc, only_connected=False, initial_sl=initial_sl)
basis_2 = basis_hc_2h(depth=l_cc, only_connected=False)

L = 51
k_path = make_triangular_grid_bz(L)
k_path = k_path.T

print("----------- calculating rotational overlaps -----------------")
#rotational overlaps
disp_cut, evs = basis_1.dispersion_nmax(k_path, two_D=False, j=J, t=t, num_n=n_bands)
print(disp_cut.shape)
ops = np.empty((disp_cut.shape[0], disp_cut.shape[1], 3), dtype=complex)
print(ops.shape)
for i, k in enumerate(k_path):
    trial_state0 = basis_1.rot_trial_state(m3=0, k=k)
    trial_state1 = basis_1.rot_trial_state(m3=1, k=k)
    trial_state2 = basis_1.rot_trial_state(m3=2, k=k)
    for j, state in enumerate(evs[i,:].T):
        ol0 = np.dot(state, trial_state0.conj())
        ops[i,j, 0] = np.abs(ol0)**2
        ol1 = np.dot(state, trial_state1.conj())
        ops[i,j, 1] = np.abs(ol1)**2
        ol2 = np.dot(state, trial_state2.conj())
        ops[i,j, 2] = np.abs(ol2)**2
ops = ops/np.max(ops)
np.save(f"../results/HC/2D_rot_overlaps_sc_depth={l_sc}_t={t}_j={J}_init_sl={initial_sl}.npy", ops)


disp_cut, evs = basis_2.dispersion_nmax(k_path, two_D=False, j=j, j_perp=J_perp, t=t, num_n=n_bands)
ops = np.empty((disp_cut.shape[0], disp_cut.shape[1], 3), dtype=complex)
for i, k in enumerate(k_path):
    trial_state0 = basis_2.rot_trial_state_from_rep(m3=0, k=k, p=-1)
    trial_state1 = basis_2.rot_trial_state_from_rep(m3=1, k=k, p=-1)
    trial_state2 = basis_2.rot_trial_state_from_rep(m3=2, k=k, p=-1)
    for j, state in enumerate(evs[i,:].T):
        ol0 = np.dot(state, trial_state0.conj())
        ops[i,j, 0] = np.abs(ol0)**2
        ol1 = np.dot(state, trial_state1.conj())
        ops[i,j, 1] = np.abs(ol1)**2
        ol2 = np.dot(state, trial_state2.conj())
        ops[i,j, 2] = np.abs(ol2)**2
ops = ops/np.max(ops)
np.save(f"../results/HC/2D_rot_overlaps_cc_depth={l_cc}_t={t}_j={J}_jperp={J_perp}.npy", ops)