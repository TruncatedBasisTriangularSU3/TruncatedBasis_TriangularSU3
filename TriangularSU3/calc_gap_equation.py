#parameters k_grid symmetric
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from time import perf_counter
import matplotlib.tri as tri
import matplotlib as mpl
from tqdm import tqdm
from scipy.optimize import fsolve
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

from importlib import reload

from TriangularSU3 import SU3_1hole_triangular
from TriangularSU3 import SU3_2hole_triangular
from TriangularSU3 import SU3_1hole_triangular2
reload(SU3_2hole_triangular) 
reload(SU3_1hole_triangular2) 
reload(SU3_1hole_triangular)   
from TriangularSU3.SU3_1hole_triangular import StringBasis

from TriangularSU3 import SU3_helper_sc_cc_overlaps
reload(SU3_helper_sc_cc_overlaps)  
from TriangularSU3.SU3_helper_sc_cc_overlaps import *


calc_overlaps = True  # choose whether to calculate the overlaps or just load them from file
honeycomb = True
cc_rot_basis = True   # choose in which basis degenerate states are diagonalized  !!! For Mirror basis, the gaps are not plottet correctly !!!
symmetry = "C3" if cc_rot_basis else "sigma"  
system = 'SU2Hc' if honeycomb else 'SU3Tri'
L=71
Ly = L
Lx = L
j = 0.3
j_perp = 0.3
t = 1
connected = True
t2=0   # t' in the sc and cc Hamiltonians (dispersions, eigenstates)
t2_overlaps = 0.1   # t' in the scattering elements M only

depth_sc = 4
depth_cc = 4
l_max_sc_overlaps = 3

unit_cell = 1

# gap parameters
beta = 10**6
doping = 0.05 # doping (has to be between 0 and 1)
delta_E = -0.1 #should be negative and not smaller then the sc bandwidth 

# Define the 3 target momentum points instead of just 1
k_ccs = [
    np.array([0.0, 0.0]),                                    # Gamma
    4 * np.pi / (3 * np.sqrt(3)) * np.array([1.0, 0.0]),     # K
    4 * np.pi / (3 * np.sqrt(3)) * np.array([-1.0, 0.0]),     # K'
    np.pi/3*np.array([np.sqrt(3), 1.0])    # M
]
K_labels = ['Gamma', 'K', "K'", 'M']

# make k_grid
k_grid = make_triangular_grid_bz(L)
kx = k_grid[0, :]
ky = k_grid[1, :]
triang = tri.Triangulation(kx, ky)

k_path = k_grid.T
if calc_overlaps:
    t0 = perf_counter()
    lat_sc1 = StringBasis(depth_sc, connected, honeycomb, unit_cell=unit_cell)
    lat_sc1.matrix_el()
    sc_disp = np.zeros([1,len(k_path)])
    sc_disp, _ = lat_sc1.dispersion(k_path, two_D=False, state=0, t=t, t2=t2, j=j, j_perp=j_perp)

    print('computed 2D dispersion in {t:.3f}s'.format(t=perf_counter()-t0))
    np.save(f'../results/TRI/{system}sc_disp_L{L}_depth{depth_sc}_t{t}_t2{t2}_J{j}_Jperp{j_perp}.npy', sc_disp) #2D dispersion saved in array of shape (2, N)


    print("----------------- Calculating eigenstates -----------------")

    lat_sc1, lat_sc2, lat_cc = init_lattices(depth_sc, depth_cc, l_max_sc_overlaps, j_perp_div_j=j_perp/j, connected=True, honeycomb=honeycomb)   

    vs_sc_1 = []
    # Create lists to hold the wavefunctions and momenta for Gamma, K, K'
    vs_sc_2_list = [[] for _ in range(len(K_labels))]
    
    mom_ind = np.indices((Lx, Ly))
    k1 = k_grid
    k_shape = k1[0].shape[0]

    # Pre-calculate k2 arrays for all 3 k_cc points
    k2s = []
    for p in range(len(K_labels)):
        k2_p = np.array([np.full(k_shape, k_ccs[p][0]), np.full(k_shape, k_ccs[p][1])])
        k2s.append(k2_p - k1)
        
    print(f'depth_cc = {depth_cc}, depth_sc = {depth_sc}, l_max_sc_overlaps = {l_max_sc_overlaps}, L = {L}, computing for {K_labels}')

    v0 = np.ones((lat_sc1.basis.length,))
    print('sc basis length:', lat_sc1.basis.length)
    if lat_sc1.basis.length > 1:
        for x in range(k_shape):
            lat_sc1.compute_H(k1[:,x], j=j, t=t, j_perp=j_perp, t2=t2)
            lat_sc1.H = lat_sc1.H.astype(np.complex128)  # in complex64 the Gamma doublet is split by ~1e-6 and not detected
            if lat_sc1.basis.length <=3:
                Es, vs = np.linalg.eigh(lat_sc1.H)
            else:
                k_num = 3 if (k1[0, x] == 0 and k1[1, x] == 0) else 2
                Es, vs = eigsh(lat_sc1.H, k=k_num, which='SA', v0=v0)
                idx = np.argsort(Es)
                Es = Es[idx]
                vs = vs[:, idx]
            
            # Diagonalize 2D degenerate subspace using C3 or mirror symmetry
            if np.isclose(Es[0], Es[1], atol=1e-7):
                if symmetry == "C3":
                    U = lat_sc1.build_rot_matrix(k1[:, x])
                elif symmetry == "sigma":
                    U = lat_sc1.build_mirror_matrix(k1[:, x])
                # Take only the 2 degenerate ground states (important at Gamma where k_num=3)
                deg_basis, _ = np.linalg.qr(vs[:, :2])
                Rm_small = deg_basis.conj().T @ U @ deg_basis
                cs, states = np.linalg.eig(Rm_small)
                
                # Sort so the symmetric (+1) mirror eigenstate is always index 0
                if symmetry == "sigma":
                    sort_idx = np.argsort(-np.real(cs)) # +1 first, -1 second
                    cs = cs[sort_idx]
                    states = states[:, sort_idx]
                    
                new_basis = deg_basis @ states
                v1 = new_basis[:, 0]
            else:
                v1 = vs[:, 0]

            # Fix phase on l=0 state
            phase_l0_state = np.exp(-1j * np.angle(v1[0])) 
            v1 = v1 * phase_l0_state
            vs_sc_1.append(v1)

            # Loop over Gamma, K, K', M to get the sc2 state for each relative momentum k2s[p]
            for p in range(len(K_labels)):
                lat_sc2.compute_H(k2s[p][:, x], j=j, t=t,t2=t2, j_perp=j_perp)
                if lat_sc2.basis.length <= 3:
                    Es, vs = np.linalg.eigh(lat_sc2.H)
                else:
                    k_num = 3 if (k2s[p][0, x] == 0 and k2s[p][1, x] == 0) else 2
                    Es, vs = eigsh(lat_sc2.H, k=k_num, which='SA', v0=v0)
                    idx = np.argsort(Es)
                    Es = Es[idx]
                    vs = vs[:, idx]
                
                # Diagonalize 2D degenerate subspace using C3 or mirror symmetry
                if np.isclose(Es[0], Es[1], atol=1e-7):
                    if symmetry == "C3":
                        U = lat_sc2.build_rot_matrix(k2s[p][:, x])
                    elif symmetry == "sigma":
                        U = lat_sc2.build_mirror_matrix(k2s[p][:, x])
                    deg_basis, _ = np.linalg.qr(vs[:, :2])
                    Rm_small = deg_basis.conj().T @ U @ deg_basis
                    cs, states = np.linalg.eig(Rm_small)
                    
                    if symmetry == "sigma":
                        sort_idx = np.argsort(-np.real(cs)) # +1 first, -1 second
                        cs = cs[sort_idx]
                        states = states[:, sort_idx]
                        
                    new_basis = deg_basis @ states
                    v2 = new_basis[:, 0]
                else:
                    v2 = vs[:, 0]
                
                # Fix phase on l=0 state
                phase_l0_state = np.exp(-1j * np.angle(v2[0])) 
                v2 = v2 * phase_l0_state

                # Adding gauge phase
                theta = 2 * np.pi / 3
                R_C3 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                r_0 = -(1/2) * np.sqrt(3) * np.array([1, -np.sqrt(3)]) 
                R_0 = 1/3 * (np.eye(2) - R_C3) @ r_0
                phi_gauge = R_0[0] * k1[0, x] + R_0[1] * k1[1, x]
                v2 = v2 * np.exp(-1j * phi_gauge)
                vs_sc_2_list[p].append(v2)
    else:
        vs_sc_1 = np.ones(k_shape)
        vs_sc_2_list = [np.ones(k_shape) for _ in range(len(K_labels))]

    vs_sc_1 = np.array(vs_sc_1).reshape((k_shape, lat_sc1.basis.length))
    for p in range(len(K_labels)):
        vs_sc_2_list[p] = np.array(vs_sc_2_list[p]).reshape((k_shape, lat_sc2.basis.length))

    # ----------------- Calculate ordered CC eigensystem -----------------
    v_cc_0_list = []
    v_cc_1_list = []
    print('cc basis length:', lat_cc.basis.length)
    for p in range(len(K_labels)):
        lat_cc.compute_H(k_ccs[p], t=t, j=j, j_perp=j_perp, t2=t2)
        H_cc = lat_cc.H.astype(np.complex128)  # in complex64 the Gamma doublet is split by ~1e-5 > atol=1e-9
        if depth_cc == 1:
            Es, vs = np.linalg.eigh(H_cc.toarray())
        else:
            # Request k=3 at Gamma in case of closeness, then sort
            Es, vs = eigsh(H_cc, k=3, which='SA')
            idx = np.argsort(Es)
            Es = Es[idx]
            vs = vs[:, idx]
        
        # Diagonalize 2D degenerate subspace using C3 or mirror symmetry
        if np.isclose(Es[0], Es[1], atol=1e-8):
            if symmetry == "C3":
                U = lat_cc.build_rot_matrix(k_ccs[p])
            elif symmetry == "sigma":
                U = lat_cc.build_mirror_matrix(k_ccs[p])
                
            deg_basis, _ = np.linalg.qr(vs[:, :2])
            print(deg_basis.shape, U.shape, K_labels[p])
            Rm_small = deg_basis.conj().T @ U @ deg_basis
            cs, states = np.linalg.eig(Rm_small)
            
            # Order the cc states:
            # For "sigma": index 0 is symmetric (+1), index 1 is antisymmetric (-1)
            # For "C3": sort by phase angle of C3 eigenvalue (e^{+i 2pi/3} vs e^{-i 2pi/3})
            if symmetry == "sigma":
                sort_idx = np.argsort(-np.real(cs))
            else:
                sort_idx = np.argsort(np.angle(cs))
            cs = cs[sort_idx]
            states = states[:, sort_idx]
            
            new_basis = deg_basis @ states
            v_cc_0 = new_basis[:, 0]
            v_cc_1 = new_basis[:, 1]
            v_cc_0 /= np.linalg.norm(v_cc_0)
            v_cc_1 /= np.linalg.norm(v_cc_1)
        else:
            v_cc_0 = vs[:, 0]
            v_cc_1 = np.zeros_like(vs[:, 0]) # if not degenerate, set second eigenvector to zero

        # Fix global phase of v_cc_0 and v_cc_1:
        # Use the component with maximum amplitude among l=0 (first 3 bond states)
        # so we never divide by a zero component on the mirror plane
        idx_0 = np.argmax(np.abs(v_cc_0[:3]))
        v_cc_0 = v_cc_0 * np.exp(-1j * np.angle(v_cc_0[idx_0]))
        
        if np.linalg.norm(v_cc_1) > 0:
            idx_1 = np.argmax(np.abs(v_cc_1[:3]))
            v_cc_1 = v_cc_1 * np.exp(-1j * np.angle(v_cc_1[idx_1]))

        U_C3 = lat_cc.build_rot_matrix(k_ccs[p])
        overlap_C3 = v_cc_1.conj().T @ (U_C3 @ v_cc_0)
        if np.abs(overlap_C3) > 1e-6:
            v_cc_1 = v_cc_1 * np.exp(1j * np.angle(overlap_C3))
        
        v_cc_0_list.append(v_cc_0)
        v_cc_1_list.append(v_cc_1)

if calc_overlaps:
    print("----------------- Calculating overlaps -----------------")

    print("1) t' overlaps")
    
    from tqdm import tqdm

    l_max = lat_cc.depth
    exchange = False
    hole_1_hop = True
    hole_2_hop = True
    
    # Store overlaps in lists of arrays (one matrix for each momentum point)
    Ms0_t = [np.zeros(k_shape, dtype=complex) for _ in range(len(K_labels))]
    Ms1_t = [np.zeros(k_shape, dtype=complex) for _ in range(len(K_labels))]
    
    if honeycomb:
        total_iterations = 2*len(lat_sc1.indices1)-1
    else:
        total_iterations = len(lat_sc1.indices1)*len(lat_sc1.indices1) 
        
    with tqdm(total=total_iterations) as pbar:
        for n1 in lat_sc1.indices1:
            for n2 in lat_sc2.indices2: 
                if len(lat_sc1.bin_basis[n1]['seq']) != 0 and len(lat_sc2.bin_basis[n2]['seq']) != 0:
                    continue
                j_max = min(lat_sc1.depth+3,lat_cc.depth+2)
                j_min = -j_max
                x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])
                y = lat_sc2.find_hole_sublattice(lat_sc2.bin_basis[n2]['seq'])
                seq1 = lat_sc1.bin_basis[n1]['seq']
                seq_hole_1 = []
                if len(seq1) != 0:
                    for i, step in enumerate(seq1):
                        seq_hole_1.append([0,step[0],step[1]]) 

                seq2 = lat_sc2.bin_basis[n2]['seq']
                seq2_sum = np.sum(np.array(seq2), axis=0, dtype=int) 

                for jx in range(j_min, j_max+1):
                    for jy in range(j_min, j_max+1):
                        offset = np.array([jx, jy])
                        lat1 = lat_sc1.bin_basis[n1]['lat'].copy()
                        subl1 = lat_sc1.find_sublattice(lat_sc1.bin_basis[n1])
                        latsc1 = (lat1-subl1)%3 
                        latsc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3 
                        sitelist1 = np.array(np.argwhere(latsc1).tolist())   

                        lat2 = lat_sc2.bin_basis[n2]['lat'].copy()
                        subl2 = lat_sc2.find_sublattice(lat_sc2.bin_basis[n2])
                        latsc2 = (lat2-subl2)%3 
                        latsc2[lat_sc2.depth+1,lat_sc2.depth+1] = 3 
                        sitelist2 = np.array(np.argwhere(latsc2).tolist())
                        if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   
                            continue  
                        phys_dist = np.zeros(2)
                        dist = np.array([jx, jy])
                        phys_dist[0], phys_dist[1] = np.sqrt(3)/2*dist[0], dist[1] - 1/2*dist[0]
                        phys_dist = np.linalg.norm(phys_dist)
                        if honeycomb:
                            if phys_dist > np.sqrt(3)/2*(j_max-1)+1:
                                continue
                        else:
                            if phys_dist > j_max:
                                continue
                        if (y-x)%3 != (jx+jy)%3:
                            continue   
                        
                        if jx == 0 and jy == 0:      
                            continue    

                        too_large = False
                        if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1)) > lat_cc.depth:
                            too_large = True                

                        state, state_uncropped = add_holes(n1, n2, jx, jy, lat_sc1, lat_sc2, lat_cc, too_large=too_large)
                        if type(state_uncropped)==dict:  # too_large configs can fit into the cc box after the hop
                            found = type(state)==dict and lat_cc.basis.search(lat_cc.state_2_list_entry(state))[0]    #discard states that are in the cc basis, since they should be either sc+sc or cc not both
                            if found:
                                continue
                            
                            ### Hole 1 hops
                            for delta in [np.array([1,2]), np.array([2,1]), np.array([1,-1]), np.array([-1,-2]), np.array([-2,-1]), np.array([-1,1])]: 
                                if hole_1_hop == False:
                                    continue
                                shift2 = np.ones((2,), dtype=int)*(offset-lat_sc1.depth-1)     
                                if np.any(np.all(sitelist2 == delta - shift2, axis=1)):  
                                    continue
                                if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1-delta)) >= lat_cc.depth + 1:
                                    continue
                                test_hole_pos = state_uncropped['hole_pos'].copy()
                                test_lat = state_uncropped['lat'].copy()
                                lat_cc.triangular_Neel(D=lat_cc.depth+1)
                                sublattice = lat_cc.Neel_state_L_size[x]
                                test_lat_new = test_lat.copy()
                                test_lat_new, test_hole_pos_new = lat_cc.make_step(test_lat_new, test_hole_pos, [0,delta[0], delta[1]], D=lat_cc.depth+1)

                                test_seq_new = state_uncropped['seq']+[[0,delta[0], delta[1]]]
                                sl,_ = lat_cc.find_hole_sublattice(test_seq_new)

                                lat2 = test_lat_new.copy()
                                lat2 = (lat2-sublattice)%3
                                lat2[tuple(test_hole_pos_new[0])] = 3
                                lat2[tuple(test_hole_pos_new[1])] = 4
                                lat2[1:-1, 1:-1] = 0 
                                if not np.all(lat2 == 0):
                                    continue

                                test_lat_new = test_lat_new[1:-1, 1:-1]
                                test_hole_pos_new = [np.array(test_hole_pos_new[0]) - 1, np.array(test_hole_pos_new[1]) - 1] 
                                
                                test_state = {'lat': test_lat_new, 'hole_pos': test_hole_pos_new, 'seq': test_seq_new}
                                found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state)) 
                                if found:
                                    repr, _, m = lat_cc.is_representative[m]
                                    phys_dist = np.array(lat_cc.dist_2_phys_dist([jx, jy], seq_hole_1))
                                    phys_delta = np.zeros(2)
                                    phys_delta[0], phys_delta[1] = np.sqrt(3)/2*delta[0], delta[1] - 1/2*delta[0]
                                    
                                    # Calculate overlaps for all 3 k_cc points concurrently
                                    for p in range(len(K_labels)):
                                        dM0 = 1/np.sqrt(2) * np.conj(v_cc_0_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist - phys_delta) - 1j * np.einsum('ax,a->x', k1, phys_delta))
                                        dM1 = 1/np.sqrt(2) * np.conj(v_cc_1_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist - phys_delta) - 1j * np.einsum('ax,a->x', k1, phys_delta))
                                        if not repr:
                                            dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist - phys_delta))
                                            dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist - phys_delta))
                                        Ms0_t[p] += dM0
                                        Ms1_t[p] += dM1

                            ### Hole 2 hops
                            for delta in [np.array([1,2]), np.array([2,1]), np.array([1,-1]), np.array([-1,-2]), np.array([-2,-1]), np.array([-1,1])]: 
                                if hole_2_hop == False:
                                    continue
                                test_hole_pos = state_uncropped['hole_pos'].copy()
                                test_lat = state_uncropped['lat'].copy()
                                lat_cc.triangular_Neel(D=lat_cc.depth+1)
                                sublattice = lat_cc.Neel_state_L_size[x]
                                if np.any(np.all(sitelist1 - (lat_sc1.depth+1) == delta + offset , axis=1)):  
                                    continue
                                if np.amax(np.abs(test_hole_pos[1]+delta-lat_cc.depth-2)) > lat_cc.depth:
                                    continue
                                test_lat_new, test_hole_pos_new = lat_cc.make_step(test_lat, test_hole_pos, [1,delta[0], delta[1]], D=lat_cc.depth+1)
                                
                                lat2 = test_lat_new.copy()
                                lat2 = (lat2-sublattice)%3
                                lat2[tuple(test_hole_pos_new[0])] = 1
                                lat2[tuple(test_hole_pos_new[1])] = 1
                                lat2[1:-1, 1:-1] = 0 
                                if not np.all(lat2 == 0):
                                    continue

                                test_seq_new = state_uncropped['seq']
                                sl,_ = lat_cc.find_hole_sublattice(test_seq_new)
                                test_lat_new = test_lat_new[1:-1, 1:-1]
                                test_hole_pos_new = [np.array(test_hole_pos_new[0]) - 1, np.array(test_hole_pos_new[1]) - 1] 
                                
                                test_state = {'lat': test_lat_new, 'hole_pos': test_hole_pos_new, 'seq': test_seq_new}
                                found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state)) 
                                if found:
                                    repr, _, m = lat_cc.is_representative[m]
                                    phys_dist = np.array(lat_cc.dist_2_phys_dist([jx, jy], seq_hole_1))
                                    phys_delta = np.zeros(2)
                                    phys_delta[0], phys_delta[1] = np.sqrt(3)/2*delta[0], delta[1] - 1/2*delta[0]
                                    
                                    # Calculate overlaps for all 3 k_cc points concurrently
                                    for p in range(len(K_labels)):
                                        dM0 = 1/np.sqrt(2) * np.conj(v_cc_0_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist))
                                        dM1 = 1/np.sqrt(2) * np.conj(v_cc_1_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist))
                                        if not repr:
                                            dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist + phys_delta))
                                            dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist + phys_delta))
                                        Ms0_t[p] += dM0
                                        Ms1_t[p] += dM1
                pbar.update(1)

    # Save outputs independently for each evaluated coordinate
    for p, K_label in enumerate(K_labels):
        np.save(f'../results/TRI/Delta_k/M_t_L={L}_{system}_symm{symmetry}_ccdepth_{depth_cc}_scdepth{depth_sc}_lmaxov_{l_max_sc_overlaps}_kcc={K_label}_j{j}_jperp{j_perp}_t{t}.npy', (Ms0_t[p], Ms1_t[p]))

    print("2) J_perp overlaps")

    l_max = lat_cc.depth
    transformed_basis = transform_lattice_j_perp(lat_cc)

    # Re-initialize isolated arrays for J_perp evaluations
    Ms0_j = [np.zeros(k_shape, dtype=complex) for _ in range(len(K_labels))]
    Ms1_j = [np.zeros(k_shape, dtype=complex) for _ in range(len(K_labels))]

    total_iterations = len(lat_sc1.indices1)*len(lat_sc2.indices2)
    with tqdm(total=total_iterations) as pbar:
        for n1 in lat_sc1.indices1:
            for n2 in lat_sc2.indices2:
                
                j_min = (np.ones(2,)*(-lat_cc.depth)).astype(int)
                j_max = (np.ones(2,)*(lat_cc.depth)).astype(int)

                x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])
                y = lat_sc2.find_hole_sublattice(lat_sc2.bin_basis[n2]['seq'])

                seq1 = lat_sc1.bin_basis[n1]['seq']
                seq_hole_1 = []
                if len(seq1) != 0:
                    for i, step in enumerate(seq1):
                        seq_hole_1.append([0,step[0],step[1]]) 

                for jx in range(j_min[0], j_max[0] + 1):          
                    for jy in range(j_min[1], j_max[1] + 1):

                        offset = np.array([jx, jy])
                        lat1 = lat_sc1.bin_basis[n1]['lat'].copy()
                        subl1 = lat_sc1.find_sublattice(lat_sc1.bin_basis[n1])
                        latsc1 = (lat1-subl1)%3 
                        latsc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3 
                        sitelist1 = np.array(np.argwhere(latsc1).tolist()) 

                        lat2 = lat_sc2.bin_basis[n2]['lat'].copy()
                        subl2 = lat_sc2.find_sublattice(lat_sc2.bin_basis[n2])
                        latsc2 = (lat2-subl2)%3 
                        latsc2[lat_sc2.depth+1,lat_sc2.depth+1] = 3 
                        sitelist2 = np.array(np.argwhere(latsc2).tolist())

                        if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   
                            continue  
                
                        phys_dist = np.zeros(2)
                        dist = np.array([jx, jy])
                        phys_dist[0], phys_dist[1] = np.sqrt(3)/2*dist[0], dist[1] - 1/2*dist[0]
                        phys_dist = np.linalg.norm(phys_dist)
                        
                        if honeycomb:
                            if phys_dist > np.sqrt(3)/2*lat_cc.depth + 0.3:
                                continue
                        else:
                            if phys_dist > lat_cc.depth:
                                continue
                        if (y-x)%3 != (jx+jy)%3:
                            continue   
                        if jx == 0 and jy == 0:      
                            continue    

                        state,_ = add_holes(n1, n2, jx, jy, lat_sc1, lat_sc2, lat_cc) 
                        
                        if type(state)==dict:
                            found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))    
                        else:
                            found = False
                            
                        if not found:
                            if type(state)== dict:
                                found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
                                if found:
                                    ms = transformed_basis.list[m_t][lat_cc.L_size + 2:]
                                    for m in ms:
                                        repr, _, m = lat_cc.is_representative[m]
                                        phys_dist = np.array(lat_cc.dist_2_phys_dist([jx, jy],seq_hole_1))
                                        
                                        for p in range(len(K_labels)):
                                            dM0 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_0_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist))
                                            dM1 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_1_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist))
                                            if not repr:
                                                dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist))
                                                dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist))
                                            Ms0_j[p] += dM0
                                            Ms1_j[p] += dM1

                            ms = add_holes_j_perp(n1, n2, jx, jy, lat_sc1, lat_sc2, lat_cc)
                            for m in ms:
                                repr, _, m = lat_cc.is_representative[m]
                                phys_dist = np.array(lat_sc1.dist_2_phys_dist([jx, jy],seq_hole_1))
                                
                                for p in range(len(K_labels)):
                                    dM0 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_0_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist))
                                    dM1 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_1_list[p][m]) * vs_sc_1[:,n1] * vs_sc_2_list[p][:,n2] * np.exp(1j * np.einsum('ax,a->x', k2s[p], phys_dist))
                                    if not repr:
                                        phys_dist_repr = np.array(lat_sc1.dist_2_phys_dist([jx, jy],seq_hole_1))
                                        dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist_repr))
                                        dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2s[p], phys_dist_repr))
                                    Ms0_j[p] += dM0
                                    Ms1_j[p] += dM1
                pbar.update(1)

    # Save J outputs
    for p, K_label in enumerate(K_labels):
        np.save(f'../results/TRI/Delta_k/M_j_L={L}_{system}_symm{symmetry}_ccdepth_{depth_cc}_scdepth{depth_sc}_lmaxov_{l_max_sc_overlaps}_kcc={K_label}_j{j}_jperp{j_perp}_t{t}.npy', (Ms0_j[p], Ms1_j[p]))
        
    print("Overlap calculations complete.")

for p, K_label in enumerate(K_labels):
    Ms_j_0, Ms_j_1 = np.load(f'../results/TRI/Delta_k/M_j_L={L}_{system}_symm{symmetry}_ccdepth_{depth_cc}_scdepth{depth_sc}_lmaxov_{l_max_sc_overlaps}_kcc={K_label}_j{j}_jperp{j_perp}_t{t}.npy', allow_pickle=True)
    Ms_t2_0, Ms_t2_1 = np.load(f'../results/TRI/Delta_k/M_t_L={L}_{system}_symm{symmetry}_ccdepth_{depth_cc}_scdepth{depth_sc}_lmaxov_{l_max_sc_overlaps}_kcc={K_label}_j{j}_jperp{j_perp}_t{t}.npy', allow_pickle=True)
    sc_disp = np.load(f'../results/TRI/{system}sc_disp_L{L}_depth{depth_sc}_t{t}_t2{0}_J{j}_Jperp{j_perp}.npy', allow_pickle=True)


# print('M_t shape:', Ms_t2_0.shape, 'M_j shape:', Ms_j_0.shape, 'sc_disp shape:', sc_disp.shape, 'k_grid shape:', k_grid.shape)

    E_min = np.min(sc_disp)
    E_max = np.max(sc_disp)

    # Old code:
    # sc_bandwidth = np.max(sc_disp) - np.min(sc_disp)
    # chem_pot1 = np.min(sc_disp)+doping*sc_bandwidth
    # epsilon = sc_disp - chem_pot1 

    # New code:
    sc_bandwidth = np.max(sc_disp) - np.min(sc_disp)
    chem_pot1 = np.percentile(sc_disp, doping * 100)
    epsilon = sc_disp - chem_pot1

    M0 = j_perp * Ms_j_0  + t2_overlaps * Ms_t2_0    #M has units of energy
    M1 = j_perp * Ms_j_1  + t2_overlaps * Ms_t2_1

    V = 9*np.sqrt(3)/2 * ((L-1)/2)**2   #real space volume of the system

    g = 1/(V*abs(delta_E)) * np.sum(np.abs(M0)**2 + (np.abs(M1)**2))

    gamma0 = 1/np.sqrt(g*abs(delta_E)) * M0
    gamma1 = 1/np.sqrt(g*abs(delta_E)) * M1


    print(f'sc bandwidth: {sc_bandwidth}, chemical potential: {chem_pot1}')
    print(f'couplig strength g: {g}')

    from scipy.optimize import fsolve

    def solve_coupled_gaps(eps_k, gamma_0, gamma_1, g, beta, initial_guess=(0.1, 0.1)):
        eps_sq = eps_k**2

        def residuals(params):
            # Pin d0 to be real, d1 is complex
            d0_re = params[0]
            d1_re = params[1]
            d1_im = params[2]
            
            d0 = d0_re
            d1 = d1_re + 1j * d1_im
            
            # Correct interference term
            # |d0*g0 + d1*g1|^2 = |d0|^2|g0|^2 + |d1|^2|g1|^2 + 2*Re(d0*g0 * conj(d1*g1))
            gap_total = d0 * gamma_0 + d1 * gamma_1
            gap_total_sq = np.abs(gap_total)**2
            Ek = np.sqrt(eps_sq + gap_total_sq)

            # Numerical safety: replace Ek=0 with a tiny value to avoid 0/0
            Ek_safe = np.where(Ek < 1e-12, 1e-12, Ek)
            thermal = np.tanh(beta * Ek_safe / 2.0) / (2 * Ek_safe)

            target_d0 = g/V * np.sum(gap_total * np.conjugate(gamma_0) * thermal)  # thermal already contains 1/(2E)
            target_d1 = g/V * np.sum(gap_total * np.conjugate(gamma_1) * thermal)
            
            # We want d0 - target_d0 = 0 and d1 - target_d1 = 0
            res_d0 = d0 - target_d0
            res_d1 = d1 - target_d1
            
            return [
                res_d0.real, # d0 is real, so imag part is redundant
                res_d1.real,
                res_d1.imag
            ]

        # Initial guess: Use smaller values! 
        # If bandwidth is ~1, try Delta ~ 0.05 - 0.2
        # guess = [0.1, 0.1, 0.01] 
        guess = [initial_guess[0].real,initial_guess[1].real, initial_guess[1].imag]
        sol, info, ier, msg = fsolve(residuals, guess, full_output=True)
        
        return sol[0], sol[1] + 1j*sol[2]

    delta1, delta2 = solve_coupled_gaps(epsilon, gamma0, gamma1, g, beta, initial_guess=(1, 1))
    np.save(f'../results/TRI/Delta_k/delta_L={L}_{system}_ccdepth_{depth_cc}_scdepth{depth_sc}_lmaxov_{l_max_sc_overlaps}_j{j}_jperp{j_perp}_t{t}_kcc={K_label}.npy', (delta1, delta2))

    print('plotting all solutions for given parameters')

    if k_ccs[p][0] == 0 and k_ccs[p][1] == 0:
        Delta = np.abs(delta1)
        phase_degrees = [0, 60, 120, 180, 240, 300]
        
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        axs_flat = axs.flatten()

        for i, deg in enumerate(phase_degrees):
            delta1, delta2 = solve_coupled_gaps(epsilon, gamma0, gamma1, g, beta, initial_guess=(Delta*1, Delta*np.exp(1j*i*np.pi/3)))
            current_Delta_k = delta1 * gamma0 + delta2 * gamma1
            print(f'initial guess: {Delta*1}, {Delta*np.exp(1j*i*np.pi/3)}')
            print(f' Delta1: {delta1}, Delta2: {delta2} for angle {deg}°')
            print()
            # Grid indexing
            idx_real = i * 2
            idx_imag = i * 2 + 1
            
            # --- Plot Real Part (using coolwarm) ---
            im_abs = axs_flat[idx_real].tripcolor(triang, np.abs(current_Delta_k), 
                                                shading='gouraud', cmap='coolwarm')
            axs_flat[idx_real].set_title(fr'$|\Delta_k|$ for $\phi={deg}^\circ$')
            fig.colorbar(im_abs, ax=axs_flat[idx_real])
            
            # --- Plot Imaginary Part (using twilight) ---
            im_angle = axs_flat[idx_imag].tripcolor(triang, np.angle(current_Delta_k), 
                                                shading='gouraud', cmap='twilight')
            axs_flat[idx_imag].set_title(fr'$\angle \Delta_k$ for $\phi={deg}^\circ$')
            fig.colorbar(im_angle, ax=axs_flat[idx_imag])

        # 3. Clean up axes
        for idx, ax in enumerate(axs_flat):
            ax.set_xlabel(r'$k_x$')
            if idx % 2 == 0:
                ax.set_ylabel(r'$k_y$')

        plt.suptitle(fr'$\Delta_k$ for $Q=\Gamma$, $doping={doping}$, $\Delta E={delta_E}$, $t2={t2}$', fontsize=22)
        plt.tight_layout()

        plt.savefig(f'../results/figures/Delta_k/{system}_kcc={K_label}_allsolutions_DeltaE{delta_E}_cp{doping}_beta{beta}_t2={t2}_lsc={depth_sc}_lcc={depth_cc}_lscmax={l_max_sc_overlaps}_L={L}.pdf')

        fig, axs = plt.subplots(3, 2, figsize=(15, 20))
        axs_flat = axs.flatten()
        phase_degrees = [0, 120, 240]

        for i, deg in enumerate(phase_degrees):
            delta1, delta2 = solve_coupled_gaps(epsilon, gamma0, gamma1, g, beta, initial_guess=(Delta*1, Delta*np.exp(1j*2*i*np.pi/3)))
            print(f'initial guess: {Delta*1}, {Delta*np.exp(1j*2*i*np.pi/3)}')
            print(f' Delta1: {delta1}, Delta2: {delta2} for angle {deg}°')
            print()
            current_Delta_k = delta1 * gamma0 + delta2 * gamma1
            # Grid indexing
            idx_real = i * 2
            idx_imag = i * 2 + 1
            
            # --- Plot Real Part (using coolwarm) ---
            im_abs = axs_flat[idx_real].tripcolor(triang, np.abs(current_Delta_k), 
                                                shading='gouraud', cmap='coolwarm')
            axs_flat[idx_real].set_title(fr'$|\Delta_k|$ for $\phi={deg}^\circ$')
            fig.colorbar(im_abs, ax=axs_flat[idx_real])
            
            # --- Plot Imaginary Part (using twilight) ---
            im_angle = axs_flat[idx_imag].tripcolor(triang, np.angle(current_Delta_k), 
                                                shading='gouraud', cmap='twilight')
            axs_flat[idx_imag].set_title(fr'$\angle \Delta_k$ for $\phi={deg}^\circ$')
            fig.colorbar(im_angle, ax=axs_flat[idx_imag])

        # 3. Clean up axes
        for idx, ax in enumerate(axs_flat):
            ax.set_xlabel(r'$k_x$')
            if idx % 2 == 0:
                ax.set_ylabel(r'$k_y$')

        plt.suptitle(fr'$\Delta_k$ for $Q=\Gamma$, $doping={doping}$, $\Delta E={delta_E}$, $t2={t2}$', fontsize=22)
        plt.tight_layout()

        # 4. Save and Show
        plt.savefig(f'../results/figures/Delta_k/{system}_kcc={K_label}_abs+angle_DeltaE{delta_E}_cp{doping}_beta{beta}_t2={t2}_lsc={depth_sc}_lcc={depth_cc}_lscmax={l_max_sc_overlaps}_L={L}.pdf')

        ## --- Plot only absolute values of 3 solutions (0, 120, 240 degrees) ---
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        phase_degrees = [0, 120, 240]

        TITLE_SIZE = 28
        LABEL_SIZE = 26
        TICK_SIZE = 24

       # Determine the leftmost and rightmost bounds of your mesh
        x_min, x_max = triang.x.min(), triang.x.max()

        for i, deg in enumerate(phase_degrees):
            delta1, delta2 = solve_coupled_gaps(epsilon, gamma0, gamma1, g, beta, initial_guess=(Delta*1, Delta*np.exp(1j*2*i*np.pi/3)))
            current_Delta_k = delta1 * gamma0 + delta2 * gamma1
            
            # Added vmin=0 to force the colormap and colorbar to start at 0.0
            im_abs = axs[i].tripcolor(triang, np.abs(current_Delta_k), 
                                      shading='gouraud', cmap='coolwarm', vmin=0)
            axs[i].set_aspect('equal')
            axs[i].set_xlim(x_min, x_max)

            axs[i].xaxis.set_major_locator(MultipleLocator(2))
            axs[i].yaxis.set_major_locator(MultipleLocator(2))
            axs[i].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Attach and configure colorbar
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_abs, cax=cax)
        
        cbar.set_label(r"$|\Delta_k|$", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))

        axs[0].set_ylabel(r'$k_y$', fontsize=LABEL_SIZE)
        axs[1].tick_params(labelleft=False)
        axs[2].tick_params(labelleft=False)
        for ax in axs:
            ax.set_xlabel(r'$k_x$', fontsize=LABEL_SIZE)

        # Replaced plt.suptitle with axs[1].set_title to match the else block's padding perfectly
        axs[1].set_title(fr'$Q=\Gamma$', fontsize=TITLE_SIZE)
        
        plt.tight_layout()

        # 4. Save and Show
        plt.savefig(f'../results/figures/Delta_k/{system}_kcc={K_label}_abs_DeltaE{delta_E}_cp{doping}_beta{beta}_t2={t2}_lsc={depth_sc}_lcc={depth_cc}_lscmax={l_max_sc_overlaps}_L={L}.pdf')

    else:
        Delta_k = delta1 * gamma0 + delta2 * gamma1
        fig, ax = plt.subplots(figsize=(5, 5))
        x_min, x_max = triang.x.min(), triang.x.max()

        # Added vmin=0 here as well
        im_abs = ax.tripcolor(triang, np.abs(Delta_k), 
                              shading='gouraud', cmap='coolwarm', vmin=0)
        
        ax.set_title(fr'$Q={K_label}$', fontsize=TITLE_SIZE)
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        
        ax.set_xlabel(r'$k_x$', fontsize=LABEL_SIZE)
        ax.set_ylabel(r'$k_y$', fontsize=LABEL_SIZE)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_abs, cax=cax)
        cbar.set_label(r"$|\Delta_k|$", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))

    plt.savefig(f'../results/figures/Delta_k/{system}_kcc={K_label}_DeltaE{delta_E}_cp{doping}_beta{beta}_t2={t2}_lsc={depth_sc}_lcc={depth_cc}_lscmax={l_max_sc_overlaps}_L={L}.pdf')
print('------------- finished plotting -------------')

#### fix Q=gamma third plot size, remove x axis numbers for second and third plot