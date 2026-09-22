import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from TriangularSU3 import SU3_1hole_triangular
reload(SU3_1hole_triangular)
from TriangularSU3.SU3_1hole_triangular import StringBasis

def compute_spectral_function(depth, t, t2, j, j_perp, N_eigenstates, eta, honeycomb=True, omega_min=-3.5, omega_max=0.5):

    print("Initializing StringBasis...")
    # Initialize the basis using the exact class from SU3_1hole_triangular.py
    sb = StringBasis(depth=depth, only_connected=True, honeycomb=honeycomb, unit_cell=0)
    
    # In the string basis, the bare hole is represented by an empty sequence
    # We find the index of the seq=[] state in the ordered bin_basis
    bare_hole_idx = None
    for i, state in enumerate(sb.bin_basis):
        if len(state['seq']) == 0:
            bare_hole_idx = i
            break
            
    if bare_hole_idx is None:
        raise ValueError("Could not find the bare hole state (seq=[]) in the basis.")
        
    print(f"Bare hole (seq=[]) found at index {bare_hole_idx}.")

    # Set up 1D Momentum Path (Gamma -> K -> M -> Gamma) 
    points_1D = 180
    K = 4*np.pi/(3*np.sqrt(3))*np.array([1, 0])
    Kp = 2*np.pi/(3*np.sqrt(3))*np.array([1, np.sqrt(3)])
    M = np.pi/3*np.array([np.sqrt(3), 1])
    Gamma = np.array([0,0])

    path1 = np.linspace(Gamma, K, int(points_1D/3), endpoint=False)
    path2 = np.linspace(K, M, int(points_1D/6), endpoint=False)
    path3 = np.linspace(M, Kp, int(points_1D/6), endpoint=False)
    path4 = np.linspace(Kp, Gamma, int(points_1D/3)+1)
    k_path = np.vstack((path1, path2, path3, path4))
    
    num_k_points = len(k_path)
    
    # Create energy grid for the spectral function plot
    omega_vals = np.linspace(omega_min, omega_max, 400) 
    spectral_map = np.zeros((len(omega_vals), num_k_points))
    
    print("Calculating eigenstates and spectral weights across k-path...")
    
    # Loop over momentum points
    for k_idx, k in enumerate(k_path):
        # Compute the Hamiltonian for the current k-point
        sb.compute_H(k, t=t, t2=t2, j=j, j_perp=j_perp)
        
        # Extract the lowest N_eigenstates eigenvalues (Es) and eigenvectors (vs)
        # Using full=True to ensure we get both eigenvalues and eigenvectors
        Es, vs = sb.eigensys(N_eigenstates - 1, full=True) 
        
        # For each energy level, extract the spectral weight and add the Lorentzian
        for n in range(len(Es)):
            energy = Es[n]
            # Spectral weight is the square of the bare hole projection
            # vs has shape (len(basis), num_states), so vs[:, n] is the nth eigenvector
            weight = np.abs(vs[bare_hole_idx, n])**2
            
            # Convolute with Lorentzian and add to the spectral map
            lorentzian = (1 / np.pi) * (eta / ((omega_vals - energy)**2 + eta**2))
            spectral_map[:, k_idx] += weight * lorentzian

    # --- Plotting the Spectral Function ---
    plt.figure(figsize=(8, 6))
    
    # Create x-axis values corresponding to k-path indices
    k_indices = np.arange(num_k_points)
    
    # Plot as a 2D color map
    plt.pcolormesh(k_indices, omega_vals, spectral_map, shading='auto', cmap='magma')
    plt.colorbar(label='Spectral Weight $A(\mathbf{k}, \omega)$')
    
    # Set symmetry point labels on the x-axis
    ticks = [0, len(path1), len(path1)+len(path2), len(path1)+len(path2)+len(path3), num_k_points-1]
    labels = ['$\Gamma$', '$K$', '$M$', '$K\'$', '$\Gamma$']
    plt.xticks(ticks, labels)
    
    # Add vertical lines at high-symmetry points
    for t_idx in ticks:
        plt.axvline(x=t_idx, color='white', linestyle='--', alpha=0.3)
        
    plt.title(f'Spinon Chargon Spectral Function (Depth={depth})')
    plt.xlabel('Momentum path')
    plt.ylabel('Energy $\omega / t$')
    plt.tight_layout()
    # plt.show()
    system = 'SU2Hc_tri_grid' if honeycomb else 'SU3Tri_tri_grid'
    path_data = "/Users/linushein/Documents/SU2_HC_Paper/graphics/"
    plt.savefig(f'{path_data}spec_func_depth{depth}_t{t}_j{j}_{system}.png', dpi=300)

if __name__ == "__main__":
    depth = 8
    t = 1.0
    t2 = 0.0
    j = 0.3
    j_perp = 0.3
    N_eigenstates = 200  # Number of lowest states to calculate
    eta = 0.05         # Lorentzian broadening factor
    honeycomb = True
    omega_min = -3.5
    omega_max = 0.5
    compute_spectral_function(depth=depth, t=t, t2=t2, j=j, j_perp=j_perp, N_eigenstates=N_eigenstates, eta=eta, honeycomb=honeycomb, omega_min=omega_min, omega_max=omega_max)