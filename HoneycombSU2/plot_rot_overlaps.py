import numpy as np
import matplotlib.pyplot as plt
import sys

#set system parameters
l_sc = 6
initial_sl = 0
l_cc = 6

t = 1
j = 0.3
j_perp = 0.3

system = 'HC' #'Tri'
twoD = True

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
xlabels = ['$\Gamma$', 'K', 'M', "K'", '$\Gamma$']

#load overlaps
ops_sc = np.load(f"../results/{system}/rot_overlaps_sc_depth={l_sc}_t={t}_j={j}_init_sl={initial_sl}.npy")
ops_cc = np.load(f"../results/{system}/rot_overlaps_cc_depth={l_cc}_t={t}_j={j}_jperp={j_perp}.npy")

disp_sc = np.load(f"../results/{system}/1D_dispersion_sc_path_GKMKpG_depth={l_sc}_t={t}_j={j}_init_sl={initial_sl}.npy")
disp_cc = np.load(f"../results/{system}/1D_dispersion_cc_path_GKMKpG_depth={l_cc}_t={t}_j={j}.npy")
x1 = np.repeat(x1, disp_sc.shape[0])

fig, axs = plt.subplots(1,3, figsize=(15,3.5))
for i in range(3):
    axs[i].scatter(x1, disp_sc.T, marker='o', alpha=np.real(ops_sc[:,:,i]))
    axs[i].set_title('$m_3=$'+str(i), size=16)
    axs[i].grid()
    axs[i].set_xticks(xticks, xlabels, size=14)
axs[0].set_ylabel('$E_0/t$', size=16)
plt.savefig(f'../results/figures/{system}_rot_overlaps_sc_depth={l_sc}_t={t}_j={j}.pdf', bbox_inches='tight')

fig, axs = plt.subplots(1,3, figsize=(15,3.5))
for i in range(3):
    axs[i].scatter(x1, disp_cc.T, marker='o', alpha=np.real(ops_cc[:,:,i]))
    axs[i].set_title('$m_3=$'+str(i), size=16)
    axs[i].grid()
    axs[i].set_xticks(xticks, xlabels, size=14)
axs[0].set_ylabel('$E_0/t$', size=16)
plt.savefig(f'../results/figures/{system}_rot_overlaps_cc_depth={l_cc}_t={t}_j={j}_jperp={j_perp}.pdf', bbox_inches='tight')

if twoD:
    
print("finished plotting")