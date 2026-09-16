import numpy as np
import matplotlib.pyplot as plt
import sys

#set system parameters
l_sc = 5
l_cc = 5

t = 1
t2 = 0
j = 0.3
j_perp = 0.3
honeycomb = True
unit_cell = 1


Code = 'TRI'
system = 'SU2Hc' if honeycomb else 'SU3Tri'


xlim = (-3.3,3.3) #this needs to match the limits in dispersions.py file
ylim = xlim
res = 0.2
karray = np.asarray(np.meshgrid(np.arange(xlim[0],xlim[1]+res,res),np.arange(ylim[0],ylim[1]+res,res)))
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

#load 2D dispersions
print("loading & plotting 2D dispersions ...")
disp_sc = np.load(f"../results/{Code}/{system}_2D_dispersion_sc_depth={l_sc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")
disp_cc = np.load(f"../results/{Code}/{system}_2D_dispersion_cc_depth={l_sc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")

#plot dispersions
fig, axs = plt.subplots(1,2, figsize=(15,5))

d1 = axs[0].imshow(disp_sc, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], cmap='coolwarm')
cb = fig.colorbar(d1, ax=axs[0], orientation='vertical', pad=0.01)
cb.set_label('$E_0/t$', size=15)
vertices_x = np.array([2*np.pi/(3*np.sqrt(3)),-2*np.pi/(3*np.sqrt(3)),-4*np.pi/(3*np.sqrt(3)), -2*np.pi/(3*np.sqrt(3)),2*np.pi/(3*np.sqrt(3)),4*np.pi/(3*np.sqrt(3)),2*np.pi/(3*np.sqrt(3))])
vertices_y = np.array([2*np.pi/3, 2*np.pi/3,0, -2*np.pi/3, -2*np.pi/3,0,2*np.pi/3])

axs[0].plot(vertices_x, vertices_y, lw=1, label='1st BZ Boundary', color='white', linestyle='--')
axs[0].set_xlabel('$k_x$', size=14)
axs[0].set_ylabel('$k_y$', size=14)
# cut_x = np.array([0,0,2*np.pi/(3*np.sqrt(3)),0])
# cut_y = np.array([0,2*np.pi/3,2*np.pi/3,0])
# axs[0].plot(cut_x, cut_y, lw=1, linestyle='--', color='black')
axs[0].plot(k_path[:,0], k_path[:,1], lw=1, color='black', linestyle='--')
axs[0].annotate('$\Gamma$', (Gamma[0], Gamma[0]), xytext=(0, -0.3), size=15)
axs[0].annotate('K', (K[0], K[1]), size=15)
axs[0].annotate("K'", (Kp[0], Kp[1]), size=15)
axs[0].annotate('M', (M[0], M[1]), size=15)
axs[0].set_title('sc', size=14)

d2 = axs[1].imshow(disp_cc, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], cmap='coolwarm')
cb = fig.colorbar(d2, ax=axs[1], orientation='vertical', pad=0.01)
cb.set_label('$E_0/t$', size=15)
vertices_x = np.array([2*np.pi/(3*np.sqrt(3)),-2*np.pi/(3*np.sqrt(3)),-4*np.pi/(3*np.sqrt(3)), -2*np.pi/(3*np.sqrt(3)),2*np.pi/(3*np.sqrt(3)),4*np.pi/(3*np.sqrt(3)),2*np.pi/(3*np.sqrt(3))])
vertices_y = np.array([2*np.pi/3, 2*np.pi/3,0, -2*np.pi/3, -2*np.pi/3,0,2*np.pi/3])

axs[1].plot(vertices_x, vertices_y, lw=1, label='1st BZ Boundary', color='white', linestyle='--')
axs[1].plot(k_path[:,0], k_path[:,1], lw=1, color='black', linestyle='--')
axs[1].set_xlabel('$k_x$', size=14)
axs[1].set_ylabel('$k_y$', size=14)
#cut_x = np.array([0,0,2*np.pi/(3*np.sqrt(3)),0])
#cut_y = np.array([0,2*np.pi/3,2*np.pi/3,0])
#axs[1].plot(cut_x, cut_y, lw=1, linestyle='--', color='black')
axs[1].annotate('$\Gamma$', (Gamma[0], Gamma[0]), xytext=(0, -0.3), size=15)
axs[1].annotate('K', (K[0], K[1]), size=15)
axs[1].annotate("K'", (Kp[0], Kp[1]), size=15)
axs[1].annotate('M', (M[0], M[1]), size=15)
axs[1].set_title('cc', size=14)
plt.savefig(f'../results/figures/{Code}_{system}_dispersions_full_BZ_sc_depth={l_sc}_cc_depth={l_cc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.pdf', bbox_inches='tight')

#plot band structure
print("loading & plotting 1D bandstructure ...")
disp_sc = np.load(f"../results/{Code}/{system}_1D_dispersion_sc_path_GKMKpG_depth={l_sc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")
disp_cc = np.load(f"../results/{Code}/{system}_1D_dispersion_cc_path_GKMKpG_depth={l_cc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.npy")
x1 = np.repeat(x1, disp_sc.shape[0]).reshape(x1.shape[0], disp_sc.shape[0])

fig, axs = plt.subplots(1,2,  figsize=(10,5))
axs[0].scatter(x1, disp_sc.T, marker='o')
axs[1].scatter(x1, disp_cc.T, marker='o')
axs[0].set_ylabel('$E_k/t$', size=16)
axs[0].grid()
axs[0].set_xticks(xticks, xlabels, size=16)
axs[1].grid()
axs[1].set_xticks(xticks, xlabels, size=16)

plt.savefig(f'../results/figures/{Code}_{system}_1D_bandstructure_GKMKpG_sc_depth={l_sc}_cc_depth={l_cc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.pdf', bbox_inches='tight')
print(f'../results/figures/{Code}_{system}_1D_bandstructure_GKMKpG_sc_depth={l_sc}_cc_depth={l_cc}_t={t}_t2={t2}_j={j}_uc={unit_cell}.pdf')


print("----------- finished plotting -----------------")