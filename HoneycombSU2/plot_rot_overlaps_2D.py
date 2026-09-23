#check if rot overlaps are C3 symmetric from Annika Code


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from TriangularSU3 import SU3_helper_sc_cc_overlaps
from TriangularSU3.SU3_helper_sc_cc_overlaps import *

k_path = make_triangular_grid_bz(51)
k_path = k_path.T
kx = k_path[:, 0]
ky = k_path[:, 1]
l_sc = 5
l_cc = l_sc
t=1
j=0.3
j_perp = 0.3
initial_sl = 0
ops_sc = np.load(f"../results/HC/2D_rot_overlaps_sc_depth={l_sc}_t={t}_j={j}_init_sl={initial_sl}.npy")
ops_cc = np.load(f"../results/HC/2D_rot_overlaps_cc_depth={l_sc}_t={t}_j={j}_jperp={j_perp}.npy")
print(ops_sc.shape)

# choose specific band from 0-5
band = 0

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i in range(3):
    weights = np.real(ops_sc[:, band, i])
    
    # Create a triangulation of your k-points
    triang = tri.Triangulation(kx, ky)
    
    # Plot the "contour" using the weights for the color/alpha effect
    # We use a single color colormap (e.g., Blues) where 0 is white/transparent
    tpc = axs[i].tripcolor(triang, weights, shading='gouraud', cmap='coolwarm')
    
    axs[i].set_title(f'$m_3={i}$', size=16)
    axs[i].set_xlabel('$k_x$', size=14)
    axs[i].set_ylabel('$k_y$', size=14)
    axs[i].set_aspect('equal')
    plt.colorbar(tpc, ax=axs[i])

plt.suptitle(fr'2D rotational eigenstate weights of sc band: {band}', fontsize=22)
plt.tight_layout()
plt.savefig(f'../results/figures/rot_overlaps_sc_2D_band{band}_depth={l_cc}_t={t}_j={j}.pdf', bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i in range(3):
    weights = np.real(ops_cc[:, band, i])
    
    # Create a triangulation of your k-points
    triang = tri.Triangulation(kx, ky)
    
    # Plot the "contour" using the weights for the color/alpha effect
    # We use a single color colormap (e.g., Blues) where 0 is white/transparent
    tpc = axs[i].tripcolor(triang, weights, shading='gouraud', cmap='coolwarm')
    
    axs[i].set_title(f'$m_3={i}$', size=16)
    axs[i].set_xlabel('$k_x$', size=14)
    axs[i].set_ylabel('$k_y$', size=14)
    axs[i].set_aspect('equal')
    plt.colorbar(tpc, ax=axs[i])
plt.suptitle(fr'2D rotational eigenstate weights of cc band: {band}', fontsize=22)
plt.tight_layout()
plt.savefig(f'../results/figures/rot_overlaps_cc_2D_band{band}_depth={l_cc}_t={t}_j={j}.pdf', bbox_inches='tight')