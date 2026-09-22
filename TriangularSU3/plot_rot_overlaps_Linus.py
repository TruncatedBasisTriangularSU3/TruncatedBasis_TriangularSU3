import numpy as np
import matplotlib.pyplot as plt
import sys

#set system parameters
l_sc = 6
l_cc = 6

t = 1
j = 0.3
t2 = 0

honeycomb = True
unit_cell = 1

Code = 'TRI'
system = 'SU2Hc' if honeycomb else 'SU3Tri'

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
# path_data = "/Users/linushein/Documents/Python/TruncatedBasis_TriangularSU3/"
path_data = "../"
ops_sc = np.load(f"{path_data}results/{Code}/{system}_rot_overlaps_sc_depth={l_sc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")
ops_cc = np.load(f"{path_data}results/{Code}/{system}_rot_overlaps_cc_depth={l_cc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")

disp_sc = np.load(f"{path_data}results/{Code}/{system}_1D_dispersion_sc_path_GKMKpG_depth={l_sc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")
disp_cc = np.load(f"{path_data}results/{Code}/{system}_1D_dispersion_cc_path_GKMKpG_depth={l_cc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")
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