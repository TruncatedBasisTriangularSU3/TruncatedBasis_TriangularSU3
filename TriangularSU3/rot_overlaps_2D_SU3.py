import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from importlib import reload

from TriangularSU3 import SU3_1hole_triangular
from TriangularSU3 import SU3_2hole_triangular
reload(SU3_1hole_triangular)
reload(SU3_2hole_triangular)
from SU3_1hole_triangular import StringBasis as basis_tri_1h
from TriangularSU3.SU3_2hole_triangular import StringBasis as basis_tri_2h
from TriangularSU3 import SU3_helper_sc_cc_overlaps
reload(SU3_helper_sc_cc_overlaps)
from TriangularSU3.SU3_helper_sc_cc_overlaps import *

#set system parameters
l_sc = 5
initial_sl = 0
l_cc = 5
n_bands = 1
unit_cell = 1
honeycomb = True
system = 'SU2Hc' if honeycomb else 'SU3Tri'

t = 1
J = 0.3
J_perp = 0.3

#define Krylov basis
basis_1 = basis_tri_1h(depth=l_sc, only_connected=False, honeycomb=honeycomb, unit_cell=unit_cell)
basis_2 = basis_tri_2h(depth=l_cc, only_connected=False, honeycomb=honeycomb, unit_cell=unit_cell)

L = 51
k_path = make_triangular_grid_bz(L)
k_path = k_path.T

print("----------- calculating rotational overlaps 1hole -----------------")
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
np.save(f"../results/TRI/{system}_2D_rot_overlaps_sc_depth={l_sc}_t={t}_j={J}_uc={unit_cell}.npy", ops)

print("----------- calculating rotational overlaps 2holes -----------------")

disp_cut, evs = basis_2.dispersion_nmax(k_path, two_D=False, j=j, j_perp=J_perp, t=t, num_n=n_bands)
ops = np.empty((disp_cut.shape[0], disp_cut.shape[1], 3), dtype=complex)
for i, k in enumerate(k_path):
    trial_state0 = basis_2.rot_trial_state(m3=0, k=k, p=-1)
    trial_state1 = basis_2.rot_trial_state(m3=1, k=k, p=-1)
    trial_state2 = basis_2.rot_trial_state(m3=2, k=k, p=-1)
    for j, state in enumerate(evs[i,:].T):
        ol0 = np.dot(state, trial_state0.conj())
        ops[i,j, 0] = np.abs(ol0)**2
        ol1 = np.dot(state, trial_state1.conj())
        ops[i,j, 1] = np.abs(ol1)**2
        ol2 = np.dot(state, trial_state2.conj())
        ops[i,j, 2] = np.abs(ol2)**2
ops = ops/np.max(ops)
np.save(f"../results/TRI/{system}_2D_rot_overlaps_cc_depth={l_cc}_t={t}_j={J}_uc={unit_cell}.npy", ops)
print("----------- finished calculations -----------------")
