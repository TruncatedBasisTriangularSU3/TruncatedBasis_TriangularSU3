import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from importlib import reload

from TriangularSU3 import SU3_1hole_triangular
from TriangularSU3 import SU3_2hole_triangular
reload(SU3_1hole_triangular)
reload(SU3_2hole_triangular)
from TriangularSU3.SU3_1hole_triangular import StringBasis as basis_tri_1h
from TriangularSU3.SU3_2hole_triangular import StringBasis as basis_tri_2h


#set system parameters
l_sc = 5
l_cc = 5
n_bands = 2

t = 1
t2 = 0
J = 0.3
unit_cell = 1
honeycomb = True
system = 'SU2Hc' if honeycomb else 'SU3Tri'
#define Krylov basis
print("params: l_sc=", l_sc, "l_cc=", l_cc, "t=", t,"t2=", t2, "J=", J, "honeycomb=", honeycomb, "unit_cell=", unit_cell)
basis_1 = basis_tri_1h(depth=l_sc, only_connected=False, honeycomb=honeycomb, unit_cell=unit_cell)
basis_2 = basis_tri_2h(depth=l_cc, only_connected=False, honeycomb=honeycomb, unit_cell=unit_cell)


#choose k array
xlim = (-3.3,3.3)
ylim = xlim
res = 0.1
karray = np.asarray(np.meshgrid(np.arange(xlim[0],xlim[1]+res,res),np.arange(ylim[0],ylim[1]+res,res)))

#relevant high symmetry points
K = 4*np.pi/(3*np.sqrt(3))*np.array([1, 0])
Kp = 2*np.pi/(3*np.sqrt(3))*np.array([1, np.sqrt(3)])
M = np.pi/3*np.array([np.sqrt(3), 1])
Gamma = np.array([0,0])

# Paths between symmetry point 
points_1D=180  
path1 = np.linspace(Gamma, K, int(points_1D/3), endpoint=False)
path2 = np.linspace(K, M, int(points_1D/6), endpoint=False)
path3 = np.linspace(M, Kp, int(points_1D/6), endpoint=False)
path4 = np.linspace(Kp, Gamma, int(points_1D/3)+1)
k_path = np.vstack((path1, path2, path3, path4))
x1 = np.linspace(0,points_1D,k_path.shape[0])

print("---------- calculating single hole disperion ------------")
disp_1,_ = basis_1.dispersion(k_array=karray, two_D=True, j=J, t=t, t2=t2)
print("---------- calculating two hole disperion ------------")
disp_2,_ = basis_2.dispersion(k_array=karray, two_D=True, j=J, t=t, t2=t2)

np.save(f"../results/TRI/{system}_2D_dispersion_cc_depth={l_cc}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_2)
np.save(f"../results/TRI/{system}_2D_dispersion_sc_depth={l_cc}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_1)

print("----------- calculating ", n_bands, " bands -----------------")
disp_bands_sc = []
disp_bands_cc = []
for n in range(n_bands):
    disp_cut_sc,_ = basis_1.dispersion(k_path, two_D=False, j=J, t=t, t2=t2, state=n)
    disp_cut_cc,_ = basis_2.dispersion(k_path, two_D=False, j=J, t=t, t2=t2, state=n)
    disp_bands_sc.append(disp_cut_sc)
    disp_bands_cc.append(disp_cut_cc)
disp_all_sc = np.array(disp_bands_sc)
disp_all_cc = np.array(disp_bands_cc)
np.save(f"../results/TRI/{system}_1D_dispersion_sc_path_GKMKpG_depth={l_sc}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_bands_sc)
np.save(f"../results/TRI/{system}_1D_dispersion_cc_path_GKMKpG_depth={l_cc}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", disp_bands_cc)

# print("----------- calculating rotational overlaps -----------------")
# #rotational overlaps
# disp_cut, evs = basis_1.dispersion_nmax(k_path, two_D=False, j=J, t=t, t2=t2, num_n=n_bands)
# ops = np.empty((disp_cut.shape[0], disp_cut.shape[1], 3), dtype=complex)
# for i, k in enumerate(k_path):
#     trial_state0 = basis_1.rot_trial_state(m3=0, k=k)
#     trial_state1 = basis_1.rot_trial_state(m3=1, k=k)
#     trial_state2 = basis_1.rot_trial_state(m3=2, k=k)
#     for j, state in enumerate(evs[i,:].T):
#         ol0 = np.dot(state, trial_state0.conj())
#         ops[i,j, 0] = np.abs(ol0)**2
#         ol1 = np.dot(state, trial_state1.conj())
#         ops[i,j, 1] = np.abs(ol1)**2
#         ol2 = np.dot(state, trial_state2.conj())
#         ops[i,j, 2] = np.abs(ol2)**2
# ops = ops/np.max(ops)
# np.save(f"../results/TRI/{system}_rot_overlaps_sc_depth={l_sc}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", ops)


# disp_cut, evs = basis_2.dispersion_nmax(k_path, two_D=False, j=J, t=t, t2=t2, num_n=n_bands)
# ops = np.empty((disp_cut.shape[0], disp_cut.shape[1], 3), dtype=complex)
# for i, k in enumerate(k_path):
#     trial_state0 = basis_2.rot_trial_state(m3=0, k=k, p=-1)
#     trial_state1 = basis_2.rot_trial_state(m3=1, k=k, p=-1)
#     trial_state2 = basis_2.rot_trial_state(m3=2, k=k, p=-1)
#     for j, state in enumerate(evs[i,:].T):
#         ol0 = np.dot(state, trial_state0.conj())
#         ops[i,j, 0] = np.abs(ol0)**2
#         ol1 = np.dot(state, trial_state1.conj())
#         ops[i,j, 1] = np.abs(ol1)**2
#         ol2 = np.dot(state, trial_state2.conj())
#         ops[i,j, 2] = np.abs(ol2)**2
# ops = ops/np.max(ops)
# np.save(f"../results/TRI/{system}_rot_overlaps_cc_depth={l_cc}_t={t}_t2={t2}_j={J}_uc={unit_cell}.npy", ops)


# print("----------- calculating 2D rotational overlaps -----------------")

# k_flat = karray.reshape(2, -1).T 

# disp_cut, evs = basis_1.dispersion_nmax(karray, two_D=True, j=J, t=t)

# # Reshape evs to easily access state vectors sequentially: (N_grid_points, N_bands, N_basis)
# evs_flat = evs.reshape(-1, evs.shape[-2], evs.shape[-1])
# ops = np.empty((evs_flat.shape[0], evs_flat.shape[1], 3), dtype=complex)

# for i, k in enumerate(k_flat):
#     trial_state0 = basis_1.rot_trial_state(m3=0, k=k)
#     trial_state1 = basis_1.rot_trial_state(m3=1, k=k)
#     trial_state2 = basis_1.rot_trial_state(m3=2, k=k)
    
#     for j, state in enumerate(evs_flat[i].T):
#         ops[i, j, 0] = np.abs(np.dot(state, trial_state0.conj()))**2
#         ops[i, j, 1] = np.abs(np.dot(state, trial_state1.conj()))**2
#         ops[i, j, 2] = np.abs(np.dot(state, trial_state2.conj()))**2

# ops = ops / np.max(ops)
# # Reshape ops back to match the original grid dimensions before saving
# ops_grid = ops.reshape(karray.shape[1], karray.shape[2], -1, 3)
# np.save(f"../results/TRI/{system}/2D_rot_overlaps_sc_depth={l_sc}_t={t}_j={J}_uc={unit_cell}.npy", ops_grid)

# print("----------- calculating 2D rotational overlaps -----------------")
# disp_cut, evs = basis_1.dispersion_nmax(karray, two_D=True, j=J, t=t)

# # Flatten the meshgrid coordinates into a clean list of 2D vectors
# k_flat = karray.reshape(2, -1).T 

# # Flatten the grid dimensions of evs: (grid_x, grid_y, bands, basis_size) -> (grid_points, bands, basis_size)
# evs_flat = evs.reshape(-1, evs.shape[-2], evs.shape[-1])
# ops = np.empty((evs_flat.shape[0], evs_flat.shape[1], 3), dtype=complex)

# for i, k in enumerate(k_flat):
#     trial_state0 = basis_1.rot_trial_state(m3=0, k=k)
#     trial_state1 = basis_1.rot_trial_state(m3=1, k=k)
#     trial_state2 = basis_1.rot_trial_state(m3=2, k=k)
    
#     # evs_flat[i] has shape (bands, basis_size)
#     # Looping over it directly yields each band's state vector with shape (basis_size,)
#     for j, state in enumerate(evs_flat[i]):
#         ops[i, j, 0] = np.abs(np.dot(state, trial_state0.conj()))**2
#         ops[i, j, 1] = np.abs(np.dot(state, trial_state1.conj()))**2
#         ops[i, j, 2] = np.abs(np.dot(state, trial_state2.conj()))**2

# ops = ops / np.max(ops)
# # Reshape back to the original 2D grid shape (grid_x, grid_y, bands, 3)
# ops_grid = ops.reshape(karray.shape[1], karray.shape[2], -1, 3)
# np.save(f"../results/TRI/{system}2D_rot_overlaps_sc_depth={l_sc}_t={t}_j={J}_uc={unit_cell}.npy", ops_grid)
# print('Saved 2D rotational overlaps for single hole system.')

# disp_cut, evs = basis_2.dispersion_nmax(karray, two_D=True, j=J, t=t)

# evs_flat = evs.reshape(-1, evs.shape[-2], evs.shape[-1])
# ops = np.empty((evs_flat.shape[0], evs_flat.shape[1], 3), dtype=complex)

# for i, k in enumerate(k_flat):
#     trial_state0 = basis_2.rot_trial_state(m3=0, k=k)
#     trial_state1 = basis_2.rot_trial_state(m3=1, k=k)
#     trial_state2 = basis_2.rot_trial_state(m3=2, k=k)
    
#     for j, state in enumerate(evs_flat[i]):
#         ops[i, j, 0] = np.abs(np.dot(state, trial_state0.conj()))**2
#         ops[i, j, 1] = np.abs(np.dot(state, trial_state1.conj()))**2
#         ops[i, j, 2] = np.abs(np.dot(state, trial_state2.conj()))**2

# ops = ops / np.max(ops)
# # Reshape ops back to match the original grid dimensions before saving
# ops_grid = ops.reshape(karray.shape[1], karray.shape[2], -1, 3)
# np.save(f"../results/TRI/{system}2D_rot_overlaps_cc_depth={l_cc}_t={t}_j={J}_uc={unit_cell}.npy", ops_grid)

print("----------- finished calculations -----------------")
