import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

from HoneycombSU2.HC_1_hole import StringBasisHC as basis_sc
from HoneycombSU2.HC_2_holes import StringBasis as basis_cc
from HoneycombSU2.HC_2_holes import sorted_list
from HoneycombSU2.helper_sc_cc_overlaps import *

print("----------------- Setting up bases and system parameters -----------------")
# define truncated bases for single and two hole channels, single hole: two basis for two sublattices
l_sc = 4
l_cc = 4
l_max_sc_overlaps = 4

lat_sc_1 = basis_sc(depth=l_sc, only_connected=False, initial_sl=0)
lat_sc_2 = basis_sc(depth=l_sc, only_connected=False, initial_sl=1)
lat_cc = basis_cc(depth=l_cc, only_connected=False)

#define system parameters
L = 100
Lx, Ly = L, L
j = 0.3
j_perp = 0.3
t = 1.0

print("sc length:", l_sc, "cc length:", l_cc)
print("max overlaps sc states:", l_max_sc_overlaps)



#define momentum grid, define high symmetry points
mom_ind = np.indices((L,L))
mom = index2momentum(mom_ind, L)
k1 = mom

K = 4*np.pi/(3*np.sqrt(3))*np.array([1, 0])
Kp = 2*np.pi/(3*np.sqrt(3))*np.array([1, np.sqrt(3)])
K0 = np.array([0,0])

k_cc = K0
k_cc_rs = k_cc.reshape(2,1,1)
k2 = np.ones((2,L,L))*k_cc_rs - k1


indices_1 = list(np.argwhere(np.count_nonzero([x['lat'] for x in lat_sc_1.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten())
indices_2 = list(np.argwhere(np.count_nonzero([x['lat'] for x in lat_sc_2.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten())

print("----------------- Calculating eigenstates -----------------")
#get eigenstates of sc for momentum grid, eigenstates of cc at k1+k2
vs_sc_1 = []
vs_sc_2 = []
v0 = np.ones((lat_sc_1.basis.length,))
if lat_sc_1.basis.length >1:
    for x in range(L):
        for y in range(L):
            lat_sc_1.compute_H(k1[:,x,y], j=j, t=t)
            if lat_sc_1.basis.length <=3:
                Es, vs = np.linalg.eigh(lat_sc_1.H)
            else:
                Es, vs = eigsh(lat_sc_1.H, k=2, which='SA', v0=v0)
            if np.isclose(Es[0],Es[1], atol=1e-7):
                print("eigenvaule of sc 1 degenerate! Rotating to R eigenvector...")
                lat_sc_1.calc_rot_mat([x,y])
                R = lat_sc_1.rot_mat
                deg_basis = vs
                deg_basis, _ = np.linalg.qr(deg_basis)
                Rm_small = deg_basis.conj().T @ R @ deg_basis
                cs, states = np.linalg.eig(Rm_small)
                new_basis = deg_basis @ states
                v1 = new_basis[:,0]
            else:
                v1 = vs[:,0]
            #fix phase
            n0 = find_l0_state(lat_sc_1)
            v1 = v1*np.exp(-1j*np.angle(v1[0]))
            vs_sc_1.append(v1)

            lat_sc_2.compute_H(k2[:,x,y], j=j, t=t)
            Es, vs = eigsh(lat_sc_2.H, k=2, which='SA', v0=v0)
            if np.isclose(Es[0],Es[1], atol=1e-7):
                print("eigenvaule of sc 2 degenerate! Rotating to R eigenvector...")
                lat_sc_2.calc_rot_mat([x,y])
                R = lat_sc_2.rot_mat
                deg_basis = vs
                deg_basis, _ = np.linalg.qr(deg_basis)
                Rm_small = deg_basis.conj().T @ R @ deg_basis
                cs, states = np.linalg.eig(Rm_small)
                new_basis = deg_basis @ states
                v2 = new_basis[:,0]
            else:
                v2 = vs[:,0]
            #fix phase
            n0 = find_l0_state(lat_sc_2)
            v2 = v2*np.exp(-1j*np.angle(v2[0]))
            vs_sc_2.append(v2)

else:
    vs_sc_1 = np.ones((Lx,Ly)) # all ones if only l=0 state
    vs_sc_2 = np.ones((Lx,Ly))

vs_sc_1 = np.array(vs_sc_1).reshape((Lx, Ly, lat_sc_1.basis.length))
vs_sc_2 = np.array(vs_sc_2).reshape((Lx, Ly, lat_sc_2.basis.length))


lat_cc.compute_H(k_cc, t=t, j=j, j_perp=j)
H_cc = lat_cc.H
if l_cc ==1:
    vs = np.array([[1,1,-1], [1,np.exp(-1j*np.pi*2/3), -1*np.exp(-1j*4*np.pi/3)], [1,np.exp(-1j*np.pi*4/3), -1*np.exp(-1j*8*np.pi/3)]]) #define C3 eigenstates by hand for l=1
    v_cc_0 = vs[:,0]
    v_cc_1 = vs[:,1]
else:
    Es, vs = eigsh(H_cc, k=2, which='SA')
    if np.isclose(Es[0],Es[1],atol=1e-9):
        print("eigenvaule of cc degenerate! Rotating to R eigenvector...")
        lat_cc.calc_rot_mat(k_cc)
        R = lat_cc.rot_mat
        deg_basis = vs
        deg_basis, _ = np.linalg.qr(deg_basis)
        Rm_small = deg_basis.conj().T @ R @ deg_basis
        cs, states = np.linalg.eig(Rm_small)
        new_basis = deg_basis @ states
        print(new_basis.shape)
        v_cc_0 = new_basis[:,0]
        v_cc_1 = new_basis[:,1]
        v_cc_0 /= np.linalg.norm(v_cc_0)
        v_cc_1 /= np.linalg.norm(v_cc_1)
        print("rot eigenvalue 0:", np.angle(np.vdot(v_cc_0, R @ v_cc_0))/(2*np.pi)*3)
        print("rot eigenvalue 1:", np.angle(np.vdot(v_cc_1, R @ v_cc_1))/(2*np.pi)*3)
    else:
        v_cc_0 = vs[:,0]
        v_cc_1 = vs[:,1]

v_cc_0 = v_cc_0 * np.exp(-1j * np.angle(v_cc_0[0]))
v_cc_1 = v_cc_1 * np.exp(-1j * np.angle(v_cc_1[0]))


vs_sc_1 = np.array(vs_sc_1).reshape((Lx, Ly, lat_sc_1.basis.length))
vs_sc_2 = np.array(vs_sc_2).reshape((Lx, Ly, lat_sc_2.basis.length))


print("----------------- Calculating overlaps -----------------")
print("1) J_perp overlaps")
transformed_basis = transform_hc_lattice_j_perp(lat_cc)

Ms_j_0 = np.zeros((L,L)) #cc state 1
Ms_j_1 = np.zeros((L,L)) #cc state 2
total_iterations = len(indices_1)*len(indices_2)
overlap_counter_0 = 0
overlap_counter_1 = 0
with tqdm(total=total_iterations) as pbar:
    for n1 in indices_1:
        for n2 in indices_2:
            state1 = lat_sc_1.bin_basis[n1]
            state2 = lat_sc_2.bin_basis[n2]
            lat1 = state1['lat']
            lat2 = state2['lat']
            sl1 = state1['sl']
            sl2 = state2['sl']

            j_min = np.ones((2,), dtype=int)*lat_cc.depth * -1
            j_max = np.ones((2,), dtype=int)*lat_cc.depth
            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    if jx == 0 and jy == 0: #holes cannot be on same site
                        continue
                    if sl1 == sl2 and (jx + jy) % 2 != 0: #same sublattice, must be an even number of sites apart
                        continue
                    if sl1 != sl2 and (jx + jy) % 2 == 0: #different sublattice, must be an odd number of sites apart
                        continue

                    lat1[l_sc+1,l_sc+1]=True
                    lat2[l_sc+1,l_sc+1]=True
                    
                    sitelist1 = np.array(np.argwhere(lat1).tolist())
                    sitelist2 = np.array(np.argwhere(lat2).tolist())

                    offset = [jy,jx]
                    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2+offset)]):   #skips all strings that would overlap, since the resulting state is not represented correctly at the moment
                    
                       continue
                    
                    # calculate distance in hc lattice:
                    jy_hc = jy
                    if sl1 != sl2:
                        jy_hc -= 2*(sl1-0.5) #ad step within unit cell if not on same sublattice
                    jx_hc = jx
                    d_hc = np.array([jy_hc, jx_hc])
                    r1, r2 = brick_to_hc_distance(d_hc)
                    rx = np.sqrt(3)*r2 + np.sqrt(3)/2*r1
                    ry = 3/2*r1


                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state = add_holes_hc(state1, state2, np.array([jy, jx]), lat_sc_1, lat_sc_2, lat_cc)
                    if type(state)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    else:
                        found = False
                    if not found:
                        if type(state)== dict:
                            found, m_t = transformed_basis.search(lat_cc.state_2_list_entry(state))
                            if found:
                                ms = transformed_basis.list[m_t][lat_cc.L_size + 3:]
                                for m in ms:
                                    repr, _, m = lat_cc.is_representative[m]
                                    dM0 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry])))
                                    dM1 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry])))
                                    if not repr:
                                        dM0 *= -1*np.exp(-1j * np.einsum('axy,a->xy', k1+k2, np.array([rx, ry])))
                                        dM1 *= -1*np.exp(-1j * np.einsum('axy,a->xy', k1+k2, np.array([rx, ry])))

                                    Ms_j_0 = Ms_j_0 + dM0
                                    Ms_j_1 = Ms_j_1 + dM1

                                    overlap_counter_0 += 1

                        #apply S ^(-)_i S^(-)_j on r.h.s. of expectation value, i.e.
                        #SS|(sc)^2>
                        ms = add_holes_j_perp(state1, state2, np.array([jy, jx]), lat_sc_1, lat_sc_2, lat_cc)
                        for m in ms:
                            repr, _, m = lat_cc.is_representative[m]
                            dM0 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry])))
                            dM1 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry])))
                            
                            if not repr:
                                dM0 *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1+k2, np.array([rx, ry])))
                                dM1 *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1+k2, np.array([rx, ry])))

                            Ms_j_0 = Ms_j_0 + dM0
                            Ms_j_1 = Ms_j_1 + dM1

                            overlap_counter_1 += 1
            pbar.update(1)

print("# of non-zero overlaps for Jperp:", overlap_counter_0, overlap_counter_1)

np.save("../results/HC/sc_cc_overlaps/M_j_perp_depth_sc="+str(l_sc)+"_depth_cc="+str(l_cc)+"_lmax_sc_overlaps="+str(l_max_sc_overlaps)+"_jperp="+str(j_perp)+"_j="+str(j)+"_t="+str(t)+".npy", (Ms_j_0, Ms_j_1))

print("2) t' overlaps")

lat_sc_1 = basis_sc(depth=l_sc, only_connected=False, initial_sl=0)
lat_sc_2 = basis_sc(depth=l_sc, only_connected=False, initial_sl=1)
lat_cc = basis_cc(depth=l_cc, only_connected=False)


Ms0 = np.zeros((L,L), dtype=complex)
Ms1 = np.zeros((L,L), dtype=complex)
total_iterations = len(indices_1)*len(indices_2)
overlap_counter = 0

with tqdm(total=total_iterations) as pbar:
    for n1 in indices_1:
        for n2 in indices_2:
            state1 = lat_sc_1.bin_basis[n1]
            state2 = lat_sc_2.bin_basis[n2]
            lat1 = state1['lat']
            lat2 = state2['lat']
            sl1 = state1['sl']
            sl2 = state2['sl']
            len1 = np.count_nonzero(lat1)
            len2 = np.count_nonzero(lat2)

            if not (len2 == 0 or len1 == 0):
                continue

            j_min = (np.ones((2,), dtype=int)*lat_cc.depth+2) * -1
            j_max = (np.ones((2,), dtype=int)*lat_cc.depth+2)
            

            for jx in range(j_min[0], j_max[0] + 1):
                for jy in range(j_min[1], j_max[1] + 1):
                    #print("trying", jx,jy)
                    if jx == 0 and jy == 0:
                        continue
                    if sl1 == sl2 and (jx + jy) % 2 != 0: #same sublattice, must be an even number of sites apart
                        #print("sl doesn't match, continue")
                        continue
                    if sl1 != sl2 and (jx + jy) % 2 == 0: #different sublattice, must be an odd number of sites apart
                        #print("sl doesn't match, continue")
                        continue
                    #print(jx,jy)
                    jy_hc = jy
                    if sl1 != sl2:
                        jy_hc -= 2*(sl1-0.5) #ad step within unit cell if not on same sublattice
                    jx_hc = jx
                    d_hc = np.array([jy_hc, jx_hc])
                    #print("sl", sl1, sl2)
                    r1, r2 = brick_to_hc_distance(d_hc)
                    rx = np.sqrt(3)*r2 + np.sqrt(3)/2*r1
                    ry = 3/2*r1
                    

                    deltas = [np.array([-1,-1]), np.array([-1,1]), np.array([0,-2]), np.array([0,2]),np.array([1,-1]), np.array([1,1])]
                    r_deltas = [np.array([1,-1]), np.array([1,0]), np.array([0,-1]), np.array([0,1]), np.array([-1,0]), np.array([-1,1])]

                    state, state_phys = add_holes_hc_hopping(state1, state2, np.array([jy, jx]), lat_sc_1, lat_sc_2, lat_cc)
                    if type(state) == dict:
                        stringlength_prev = np.count_nonzero(lat1)+np.count_nonzero(lat2)
                        stringlength_new = np.count_nonzero(state['lat'])
                        if stringlength_new != stringlength_prev:
                            continue

                    large_depth = lat_sc_1.depth + lat_cc.depth + 2
                    
                    if type(state_phys)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state_phys))
                    else:
                        found = False
                    if not found:
                        if type(state) == dict:
                            for d, delta in enumerate(deltas):
                                
                                ## Hopping of hole 1 (w.r.t. we chose the reference frame)
                                rd = r_deltas[d] #distance in hc lattice vectors
                                rd = brick_to_hc_distance(delta)
                                rdx = np.sqrt(3)*rd[1]+np.sqrt(3)/2*rd[0]
                                rdy = 3/2*rd[0]
                                hole_pos = [state['hole_pos'][0] + delta, state['hole_pos'][1]]
                                
                                sl = state['sl']
                                test_lat = state['lat'].copy()
                                
                                pos0 = tuple(np.array([1, 1]) * (large_depth+1)) #initial position of central hole
                                pos1 = tuple(np.array([1, 1]) * (large_depth+1) + delta) #new position of central hole after hopping
                                pos2 = tuple(np.array([1, 1]) * (large_depth+1) + np.array([jy, jx])) #new position of added hole
                                test_lat[pos0] = test_lat[pos1] #spin moves to central site
                                test_lat[pos1] = False #hole positions false
                                test_lat[pos2] = False
                                
                                plot_state = {'lat': test_lat, 'hole_pos': hole_pos, 'sl': sl}
                                if not np.any(np.abs(hole_pos[0]-hole_pos[1])>lat_cc.depth+2):
                                   
                                    test_lat, hole_pos = lat_cc.translation(test_lat, hole_pos, np.concatenate(([0], delta))) #this should be independent of l_cc so ts ok taht our lattice is still larger
                                    plot_state2 = {'lat': test_lat, 'hole_pos': hole_pos, 'sl': sl}
                                    
                                    dx = large_depth - lat_cc.depth
                                    #check if any cropped sites are true
                                    string_length_big = np.count_nonzero(test_lat)
                                    test_lat = test_lat[dx:-dx,dx:-dx]
                                    string_length_small = np.count_nonzero(test_lat)
                                    if string_length_big != string_length_small:
                                        continue
                                    #plot_lat(test_lat)
                                    hole_pos = [hole_pos[0]-np.array([dx,dx]), hole_pos[1]-np.array([dx,dx])]
                                    test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'sl': sl} #sublattice not changed by nnn hopping
                                    
                                    found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
                                    if found:
                                       
                                        repr, _, m = lat_cc.is_representative[m]
                                        
                                        dM0 = 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry]) - np.array([rdx, rdy])) - 1j * np.einsum('axy,a->xy', k1, np.array([rdx, rdy])))
                                        dM1 = 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry]) - np.array([rdx, rdy])) - 1j * np.einsum('axy,a->xy', k1, np.array([rdx, rdy])))
                                        # factor 1/sqrt(2) comes from projection onto fermionic states
                                        if not repr:
                                            dM0 *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([rx, ry]) - np.array([rdx,rdy])))
                                            dM1 *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([rx, ry]) - np.array([rdx,rdy])))

                                        Ms0 = Ms0 + dM0
                                        Ms1 = Ms1 + dM1
                                        overlap_counter +=1
                                   

                                ### Hole 2 hops
                                if np.any(np.abs((np.array([jy,jx])+delta))>lat_cc.depth):
                                    #print("hole hops to far away, continue")
                                    continue
                                hole_pos = [state['hole_pos'][0], state['hole_pos'][1] + delta] 
                                test_lat = state['lat'].copy()
                                pos0 = tuple(np.array([1, 1]) * (large_depth+1)) #initial position of central hole
                                pos1 = tuple(np.array([1, 1]) * (large_depth +1) + np.array([jy, jx])) #initial position of added hole
                                pos2 = tuple(np.array([1, 1]) * (large_depth +1) + np.array([jy, jx]) + delta) #new position of added hole
                                test_lat[pos1] = test_lat[pos2] #this is flipped
                                test_lat[pos2] = False #this is now the hole
                                
                                dx = large_depth - lat_cc.depth
                                string_length_big = np.count_nonzero(test_lat)
                                test_lat = test_lat[dx:-dx,dx:-dx]
                                string_length_small = np.count_nonzero(test_lat)
                                if string_length_big != string_length_small:
                                    continue
                                hole_pos = [hole_pos[0]-np.array([dx,dx]), hole_pos[1]-np.array([dx,dx])]
                                test_state = {'lat': test_lat, 'hole_pos': hole_pos, 'sl': sl}
                                
                                found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state))
                                if found:
                                    
                                    repr, _, m = lat_cc.is_representative[m]
                                    
                                    dM0 = 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry])))
                                    dM1 = 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,:,n1] * vs_sc_2[:,:,n2] * np.exp(1j * np.einsum('axy,a->xy', k2, np.array([rx, ry])))
                                    
                                    # factor 1/sqrt(2) comes from projection onto ferminoic states
                                    if not repr:
                                        dM0 *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([rx, ry]) + np.array([rdx,rdy])))
                                        dM1 *= -1 * np.exp(-1j * np.einsum('axy,a->xy', k1 + k2, np.array([rx, ry]) + np.array([rdx,rdy])))

                                    Ms0 = Ms0 + dM0
                                    Ms1 = Ms1 + dM1
                                    overlap_counter +=1
            pbar.update(1)
    Ms_t_0 = Ms0
    Ms_t_1 = Ms1

 
    print("total overlaps t_prime:", overlap_counter)

np.save("../results/HC/sc_cc_overlaps/M_t_depth_sc="+str(l_sc)+"_depth_cc="+str(l_cc)+"_lmax_sc_overlaps="+str(l_max_sc_overlaps)+"_jperp="+str(j_perp)+"_j="+str(j)+"_t="+str(t)+".npy", (Ms_t_0, Ms_t_1))

print("Overlap calculations complete.")

fig, axs = plt.subplots(2,2, figsize=(8,7))
im1 = axs[0,0].imshow(np.abs(Ms_t_0), cmap='coolwarm', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[0,0].set_title("$|M^{t'}_1|$")
axs[0,0].set_ylabel("$k_y$")
fig.colorbar(im1, ax=axs[0,0])
im2 = axs[1,0].imshow(np.angle(Ms_t_0), cmap='twilight', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[1,0].set_ylabel("$k_y$")
axs[1,0].set_xlabel("$k_x$")
axs[1,0].set_title("phase($M^{t'}_1$)")
fig.colorbar(im2, ax=axs[1,0])
im3 = axs[0,1].imshow(np.abs(Ms_t_1), cmap='coolwarm', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[0,1].set_title("$|M^{t'}_2|$")
fig.colorbar(im3, ax=axs[0,1])
im4 = axs[1,1].imshow(np.angle(Ms_t_1), cmap='twilight', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[1,1].set_title("phase($M^{t'}_2$)")
axs[1,1].set_xlabel("$k_x$")
fig.colorbar(im4, ax=axs[1,1])
plt.savefig("../results/figures/M_tprime_depth_sc="+str(l_sc)+"_depth_cc="+str(l_cc)+"_lmax_sc_overlaps="+str(l_max_sc_overlaps)+"_jperp="+str(j_perp)+"_j="+str(j)+"_t="+str(t)+".pdf", bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(2,2, figsize=(8,7))
im1 = axs[0,0].imshow(np.abs(Ms_j_0), cmap='coolwarm', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[0,0].set_title("$|M^{J_\perp}_1|$")
axs[0,0].set_ylabel("$k_y$")
fig.colorbar(im1, ax=axs[0,0])
im2 = axs[1,0].imshow(np.angle(Ms_j_0), cmap='twilight', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[1,0].set_ylabel("$k_y$")
axs[1,0].set_xlabel("$k_x$")
axs[1,0].set_title("phase($M^{J_\perp}_1$)")
fig.colorbar(im2, ax=axs[1,0])
im3 = axs[0,1].imshow(np.abs(Ms_j_1), cmap='coolwarm', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[0,1].set_title("$|M^{J_\perp}_2|$")
fig.colorbar(im3, ax=axs[0,1])
im4 = axs[1,1].imshow(np.angle(Ms_j_1), cmap='twilight', extent=[k1[0,0,0], k1[0,-1,-1], k1[1,0,0], k1[1,-1,-1]])
axs[1,1].set_title("phase($M^{J_\perp}_2$)")
axs[1,1].set_xlabel("$k_x$")
fig.colorbar(im4, ax=axs[1,1])
plt.savefig("../results/figures/M_jperp_depth_sc="+str(l_sc)+"_depth_cc="+str(l_cc)+"_lmax_sc_overlaps="+str(l_max_sc_overlaps)+"_jperp="+str(j_perp)+"_j="+str(j)+"_t="+str(t)+".pdf", bbox_inches='tight')
plt.show()