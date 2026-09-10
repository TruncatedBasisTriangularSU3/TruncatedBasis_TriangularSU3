
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from importlib import reload

from TriangularSU3 import SU3_1hole_triangular2
from TriangularSU3 import SU3_2hole_triangular
reload(SU3_1hole_triangular2)
reload(SU3_2hole_triangular)
from TriangularSU3.SU3_1hole_triangular2 import StringBasis as basis_tri_1h2
from TriangularSU3.SU3_2hole_triangular import StringBasis as basis_tri_2h


#set system parameters
l_sc = 4
l_cc = 4
n_bands = 5

t = 1
J = 0.3
unit_cell = 1
honeycomb = False
system = 'SU2Hc2' if honeycomb else 'SU3Tri2'
#define Krylov basis
print("params: l_sc=", l_sc, "l_cc=", l_cc, "t=", t, "J=", J, "honeycomb=", honeycomb, "unit_cell=", unit_cell)
basis_1 = basis_tri_1h2(depth=l_sc, only_connected=False, honeycomb=honeycomb, unit_cell=unit_cell)
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

# print("---------- calculating single hole disperion ------------")
# disp_1,_ = basis_1.dispersion(k_array=karray, two_D=True, j=J, t=t)
# print("---------- calculating two hole disperion ------------")
# disp_2,_ = basis_2.dispersion(k_array=karray, two_D=True, j=J, t=t)

# np.save(f"../results/TRI/{system}_2D_dispersion_cc_depth={l_cc}_t={t}_j={J}_uc={unit_cell}.npy", disp_2)
# np.save(f"../results/TRI/{system}_2D_dispersion_sc_depth={l_cc}_t={t}_j={J}_uc={unit_cell}.npy", disp_1)

print("----------- calculating ", n_bands, " bands -----------------")
disp_bands_sc = []
disp_bands_cc = []
for n in range(n_bands):
    disp_cut_sc,_ = basis_1.dispersion(k_path, two_D=False, j=J, t=t, state=n)
    disp_cut_cc,_ = basis_2.dispersion(k_path, two_D=False, j=J, t=t, state=n)
    disp_bands_sc.append(disp_cut_sc)
    disp_bands_cc.append(disp_cut_cc)
disp_all_sc = np.array(disp_bands_sc)
disp_all_cc = np.array(disp_bands_cc)
np.save(f"../results/TRI/{system}_1D_dispersion_sc_path_GKMKpG_depth={l_sc}_t={t}_j={J}_uc={unit_cell}.npy", disp_bands_sc)
np.save(f"../results/TRI/{system}_1D_dispersion_cc_path_GKMKpG_depth={l_cc}_t={t}_j={J}_uc={unit_cell}.npy", disp_bands_cc)

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
np.save(f"../results/TRI/{system}_rot_overlaps_sc_depth={l_sc}_t={t}_j={J}_uc={unit_cell}.npy", ops)


disp_cut, evs = basis_2.dispersion_nmax(k_path, two_D=False, j=J, t=t, num_n=n_bands)
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
np.save(f"../results/TRI/{system}_rot_overlaps_cc_depth={l_cc}_t={t}_j={J}_uc={unit_cell}.npy", ops)

print("----------- finished calculations -----------------")


import numpy as np
import matplotlib.pyplot as plt
import sys

#set system parameters
l_sc = 4
l_cc = 4

t = 1
j = 0.3

honeycomb = False
unit_cell = 1

Code = 'TRI'

#K space values
K = 4*np.pi/(3*np.sqrt(3))*np.array([1, 0])
Kp = 2*np.pi/(3*np.sqrt(3))*np.array([1, np.sqrt(3)])
M = np.pi/3*np.array([np.sqrt(3), 1])
Gamma = np.array([0,0])
points_1D=180
# Paths between symmetry point   
path1 = np.linspace(Gamma, K, int(points_1D/3), endpoint=False)
path2 = np.linspace(K, M, int(points_1D/6), endpoint=False)
path3 = np.linspace(M, Kp, int(points_1D/6), endpoint=False)
path4 = np.linspace(Kp, Gamma, int(points_1D/3)+1)
k_path = np.vstack((path1, path2, path3, path4))
x1 = np.linspace(0,points_1D,k_path.shape[0])
xticks = [0, 60, 90, 120, 180]
xlabels = ['$\\Gamma$', 'K', 'M', "K'", '$\\Gamma$']

#load overlaps
path_data = "../"
ops_sc = np.load(f"{path_data}results/{Code}/{system}_rot_overlaps_sc_depth={l_sc}_t={t}_j={j}_uc={unit_cell}.npy")
ops_cc = np.load(f"{path_data}results/{Code}/{system}_rot_overlaps_cc_depth={l_cc}_t={t}_j={j}_uc={unit_cell}.npy")

disp_sc = np.load(f"{path_data}results/{Code}/{system}_1D_dispersion_sc_path_GKMKpG_depth={l_sc}_t={t}_j={j}_uc={unit_cell}.npy")
disp_cc = np.load(f"{path_data}results/{Code}/{system}_1D_dispersion_cc_path_GKMKpG_depth={l_cc}_t={t}_j={j}_uc={unit_cell}.npy")
x1 = np.repeat(x1, disp_sc.shape[0])
print(disp_sc.shape)
print(disp_sc.shape[0])
fig, axs = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    axs[i].scatter(x1, disp_sc.T, marker='o', alpha=np.real(ops_sc[:,:,i]))
    axs[i].set_title('$m_3=$'+str(i), size=20)
    axs[i].grid()
    axs[i].set_xticks(xticks, xlabels, size=20)
    axs[i].tick_params(axis='y', labelsize=20)  # Increase the size of the numbers on the y-axis
axs[0].set_ylabel('$E_0/t$', size=20)
# plt.suptitle(fr'sc rotational overlap for unit cell: {unit_cell}',size=18)
plt.tight_layout()
plt.savefig(f'{path_data}results/figures/{Code}_{system}_rot_overlaps_sc_depth={l_sc}_t={t}_j={j}_uc={unit_cell}.pdf', bbox_inches='tight')


print(f'disp_cc.shape: {disp_cc.shape}, ops_cc.shape: {ops_cc.shape}')
print(f' x1 shape: {x1.shape}')
fig, axs = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    axs[i].scatter(x1, disp_cc.T, marker='o', alpha=np.real(ops_cc[:,:,i]))
    axs[i].set_title('$m_3=$'+str(i), size=20)
    axs[i].grid()
    axs[i].set_xticks(xticks, xlabels, size=20)
    axs[i].tick_params(axis='y', labelsize=20)
axs[0].set_ylabel('$E_0/t$', size=20)
# plt.suptitle(fr'cc rotational overlap for unit cell: {unit_cell}',size=18)
plt.tight_layout()
plt.savefig(f'{path_data}results/figures/{Code}_{system}_rot_overlaps_cc_depth={l_cc}_t={t}_j={j}_uc={unit_cell}.pdf', bbox_inches='tight')
print("----------- finished plotting -----------------")