import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from HoneycombSU2.HC_1_hole import StringBasisHC as basis_hc_1h
from HoneycombSU2.HC_2_holes import StringBasis as basis_hc_2h


#set system parameters
l_sc = 6
initial_sl = 0
l_cc = 6
n_bands = 4

t = 1
J = 0.3
J_perp = 0.3

#define Krylov basis
basis_1 = basis_hc_1h(depth=l_sc, only_connected=False, initial_sl=initial_sl)
basis_2 = basis_hc_2h(depth=l_cc, only_connected=False)


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
disp_1 = basis_1.dispersion(k_array=karray, two_D=True, j=J, t=t)
print("---------- calculating two hole disperion ------------")
disp_2 = basis_2.dispersion(k_array=karray, two_D=True, j=J, t=t)

np.save(f"../results/HC/2D_dispersion_cc_depth={l_cc}_t={t}_j={J}_kmin={xlim[0]}_kmax={xlim[1]}.npy", disp_2)
np.save(f"../results/HC/2D_dispersion_sc_depth={l_cc}_t={t}_j={J}_kmin={xlim[0]}_kmax={xlim[1]}_init_sl={initial_sl}.npy", disp_1)

print("----------- calculating ", n_bands, " bands -----------------")
disp_bands_sc = []
disp_bands_cc = []
for n in range(n_bands):
    disp_cut_sc = basis_1.dispersion(k_path, two_D=False, j=J, t=t, state=n)
    disp_cut_cc = basis_2.dispersion(k_path, two_D=False, j=J, t=t, state=n)
    disp_bands_sc.append(disp_cut_sc)
    disp_bands_cc.append(disp_cut_cc)
disp_all_sc = np.array(disp_bands_sc)
disp_all_cc = np.array(disp_bands_cc)
np.save(f"../results/HC/1D_dispersion_sc_path_GKMKpG_depth={l_cc}_t={t}_j={J}_init_sl={initial_sl}.npy", disp_bands_sc)
np.save(f"../results/HC/1D_dispersion_cc_path_GKMKpG_depth={l_cc}_t={t}_j={J}.npy", disp_bands_cc)

print("----------- calculating rotational overlaps -----------------")
#rotational overlaps
disp_cut, evs = basis_1.dispersion_nmax(k_path, two_D=False, j=J, t=t, num_n=n_bands)
ops = np.empty((disp_cut.shape[0], disp_cut.shape[1], 3), dtype=complex)
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
np.save(f"../results/HC/rot_overlaps_sc_depth={l_sc}_t={t}_j={J}_init_sl={initial_sl}.npy", ops)


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
np.save(f"../results/HC/rot_overlaps_cc_depth={l_cc}_t={t}_j={J}_jperp={J_perp}.npy", ops)