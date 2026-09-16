from importlib import reload
import matplotlib as mpl
mpl.use('Agg')  # Prevents GUI windows from holding the terminal open
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import matplotlib as mpl
import os
from tqdm import tqdm

from TriangularSU3 import SU3_1hole_triangular
from TriangularSU3 import SU3_2hole_triangular
from TriangularSU3 import SU3_1hole_triangular2
reload(SU3_2hole_triangular) 
reload(SU3_1hole_triangular2) 
reload(SU3_1hole_triangular)   
from TriangularSU3.SU3_1hole_triangular import StringBasis as Lat_sc1
from TriangularSU3.SU3_1hole_triangular2 import StringBasis as Lat_sc2
from TriangularSU3.SU3_2hole_triangular import StringBasis as Lat_cc 

from TriangularSU3 import SU3_helper_sc_cc_overlaps
reload(SU3_helper_sc_cc_overlaps)  
from TriangularSU3.SU3_helper_sc_cc_overlaps import *

"decide if you want to calculate honeycomb or triangular lattice overlaps and other relevant parameters here:"
honeycomb = True
system = 'SU2Hc_tri_grid' if honeycomb else 'SU3Tri_tri_grid'
connected = True
unit_cell = 1
depth_sc = 5
depth_cc = 5
l_max_sc_overlaps = 3
L=101 # must be odd

k_grid = make_triangular_grid_bz(L, grid_size = 2*np.pi/np.sqrt(3) * L/(L-1))
k_cc = np.array([0,0])

j = 0.3
j_perp = 0.3
t = 1

print(f'depth_cc = {depth_cc}, depth_sc = {depth_sc}, l_max_sc_overlaps = {l_max_sc_overlaps}, L = {L}, k_cc = {k_cc}')

lat_sc1 = Lat_sc1(depth_sc, only_connected=connected, honeycomb=honeycomb, unit_cell=unit_cell)
lat_sc2 = Lat_sc2(depth_sc, only_connected=connected, honeycomb=honeycomb, unit_cell=unit_cell)
lat_sc1.matrix_el() 
lat_sc2.matrix_el() 

lat_sc1.indices1 = list(np.argwhere(np.array([len(x['seq']) for x in lat_sc1.bin_basis]) <= l_max_sc_overlaps).flatten()) # linus
lat_sc2.indices2 = list(np.argwhere(np.array([len(x['seq']) for x in lat_sc2.bin_basis]) <= l_max_sc_overlaps).flatten()) # linus

lat_cc = Lat_cc(depth_cc, only_connected=connected, honeycomb=honeycomb)
lat_cc.matrix_el()

print('initialized lattices')

k1 = k_grid
k_shape = k1[0].shape[0]
k2 = np.array([np.full(k_shape, k_cc[0]), np.full(k_shape, k_cc[1])])
k2 = k2 - k1

print("----------------- Calculating eigenstates -----------------")
vs_sc_1 = []
vs_sc_2 = []
v0 = np.ones((lat_sc1.basis.length,))
if lat_sc1.basis.length >1:
    for x in range(k_shape):
        lat_sc1.compute_H(k1[:,x], j=j, t=t)
        if lat_sc1.basis.length <=3:
            Es, vs = np.linalg.eigh(lat_sc1.H)
        else:
            if k1[0,x] == 0 and k1[1,x] == 0:
                # print(f'k1 = {k1[:,x,y]}')
                Es, vs = eigsh(lat_sc1.H, k=3, which='SA', v0=v0)
                idx = np.argsort(Es)
                Es = Es[idx]
                vs = vs[:, idx]
            else:
                Es, vs = eigsh(lat_sc1.H, k=2, which='SA', v0=v0)
        # print(vs)
        if np.isclose(Es[0],Es[1], atol=1e-7):
            print("eigenvaule of sc 1 degenerate! Rotating to R eigenvector...")
            R = lat_sc1.build_rot_matrix(k1[:,x])
            deg_basis = vs
            deg_basis, _ = np.linalg.qr(deg_basis)
            Rm_small = deg_basis.conj().T @ R @ deg_basis
            cs, states = np.linalg.eig(Rm_small)
            new_basis = deg_basis @ states
            v1 = new_basis[:,0]
        else:
            v1 = vs[:,0]
        #fix phase
        n0 = find_l0_state_sc1(lat_sc1)
        v1 = v1*np.exp(-1j*np.angle(v1[n0])) #changed v1[0] to v1[n0] to fix phase according to l0 state
        vs_sc_1.append(v1)

        lat_sc2.compute_H(k2[:,x], j=j, t=t)
        Es, vs = eigsh(lat_sc2.H, k=2, which='SA', v0=v0)
        if lat_sc2.basis.length <=3:
            Es, vs = np.linalg.eigh(lat_sc2.H)
        else:
            if k1[0,x] == 0 and k1[1,x] == 0:
                Es, vs = eigsh(lat_sc2.H, k=3, which='SA', v0=v0)
                idx = np.argsort(Es)
                Es = Es[idx]
                vs = vs[:, idx]
            else:
                Es, vs = eigsh(lat_sc2.H, k=2, which='SA', v0=v0)
        #print(f'sc2: for k={k1[:,x,y]} Es0 = {Es[0]}, Es1 = {Es[1]}, Es2 = {Es[2]}')
        if np.isclose(Es[0],Es[1], atol=1e-7):
            print("eigenvaule of sc 2 degenerate! Rotating to R eigenvector...")
            R = lat_sc2.build_rot_matrix(k1[:,x])
            deg_basis = vs
            deg_basis, _ = np.linalg.qr(deg_basis)
            Rm_small = deg_basis.conj().T @ R @ deg_basis
            cs, states = np.linalg.eig(Rm_small)
            new_basis = deg_basis @ states
            v2 = new_basis[:,0]
        else:
            v2 = vs[:,0]
        #fix phase
        n0 = find_l0_state_sc2(lat_sc2) 
        v2 = v2*np.exp(-1j*np.angle(v2[n0])) #changed v2[0] to v2[n0] to fix phase according to l0 state

        # adding gauge phase
        theta = 2 * np.pi / 3
        R_C3 = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),  np.cos(theta)]])
        r_0 = -(1/2)*np.sqrt(3) * np.array([1, -np.sqrt(3)]) 
        R_0 = 1/3*(np.eye(2) - R_C3) @ r_0
        phi_gauge = R_0[0]*k1[0,x] + R_0[1]*k1[1,x]

        vs_sc_2.append(v2)
else:
    vs_sc_1 = np.ones((Lx,Ly))
    vs_sc_2 = np.ones((Lx,Ly))

vs_sc_1 = np.array(vs_sc_1).reshape((k_shape, lat_sc1.basis.length))
vs_sc_2 = np.array(vs_sc_2).reshape((k_shape, lat_sc2.basis.length))

lat_cc.compute_H(k_cc, t=t, j=j, j_perp=j)
H_cc = lat_cc.H
if depth_cc ==1:
    Es, vs = np.linalg.eigh(H_cc.toarray())
else:
    Es, vs = eigsh(H_cc, k=2, which='SA')
if np.isclose(Es[0],Es[1],atol=1e-9):
    print("eigenvaule of cc degenerate! Rotating to R eigenvector...")
    R = lat_cc.build_rot_matrix(k_cc)
    deg_basis = vs
    deg_basis, _ = np.linalg.qr(deg_basis)
    Rm_small = deg_basis.conj().T @ R @ deg_basis
    cs, states = np.linalg.eig(Rm_small)
    new_basis = deg_basis @ states
    # print(new_basis.shape)
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

 
print("----------------- Calculating overlaps -----------------")
print("1) t' overlaps")

l_max = lat_cc.depth
hole_1_hop = True
hole_2_hop = True
Ms0 = np.zeros((L,L), dtype=complex)
Ms1 = np.zeros((L,L), dtype=complex)
overlap_counter = 0

Ms0 = np.zeros(k_shape)
Ms1 = np.zeros(k_shape)

overlap_count = 0
if honeycomb:
    total_iterations = 2*len(lat_sc1.indices1)-1 #since either n1 or n2 have to be zero
else:
    total_iterations = len(lat_sc1.indices1)*len(lat_sc1.indices1) #evaluate all combinations
with tqdm(total=total_iterations) as pbar:
    for n1 in lat_sc1.indices1:  #if we let hole 2 hop as well we need to have symmetrie in strings 
        for n2 in lat_sc2.indices2: 

            if honeycomb:
                if len(lat_sc1.bin_basis[n1]['seq']) != 0 and len(lat_sc2.bin_basis[n2]['seq']) != 0:
                    continue

            j_max = min(lat_sc1.depth+3,lat_cc.depth+2)
            j_min = -j_max
            # j_min = (np.ones(2,)*(-lat_cc.depth-2)).astype(int) #improve these bounds in the future
            # j_max = (np.ones(2,)*(lat_cc.depth+2)).astype(int)

            x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])
            y = lat_sc2.find_hole_sublattice(lat_sc2.bin_basis[n2]['seq'])

            # if not x==2 and not y==2:   #selects only couplings that are exclusive to triangular lattice
            #     continue

            # if x==2 or y==2:   #selects only couplings that are exclusive to honeycomb lattice
            #     continue

            seq1 = lat_sc1.bin_basis[n1]['seq']
            seq_hole_1 = []
            if len(seq1) != 0:
                for i, step in enumerate(seq1):
                    seq_hole_1.append([0,step[0],step[1]]) #bring seq_hole_1 in the form [[0,x,y],...] add to dist_2_phys_dist

            seq2 = lat_sc2.bin_basis[n2]['seq']
            seq2_sum = np.sum(np.array(seq2), axis=0, dtype=int) #total displacement of hole 2 w.r.t. origin

            for jx in range(j_min, j_max+1):
                for jy in range(j_min, j_max+1):
            # for jx in [2]:
            #     for jy in [0]:
                    offset = np.array([jx, jy])
                    lat1 = lat_sc1.bin_basis[n1]['lat'].copy()
                    subl1 = lat_sc1.find_sublattice(lat_sc1.bin_basis[n1])
                    latsc1 = (lat1-subl1)%3 
                    latsc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3 #to include hole in sitelist
                    sitelist1 = np.array(np.argwhere(latsc1).tolist())   

                    lat2 = lat_sc2.bin_basis[n2]['lat'].copy()
                    subl2 = lat_sc2.find_sublattice(lat_sc2.bin_basis[n2])
                    latsc2 = (lat2-subl2)%3 
                    latsc2[lat_sc2.depth+1,lat_sc2.depth+1] = 3 #to include hole in sitelist
                    sitelist2 = np.array(np.argwhere(latsc2).tolist())
                    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   
                        continue  #skips all strings that would overlap, since the resulting state is not represented correctly at the moment
                    phys_dist = np.zeros(2)
                    dist = np.array([jx, jy])
                    phys_dist[0], phys_dist[1] = np.sqrt(3)/2*dist[0], dist[1] - 1/2*dist[0]
                    phys_dist = np.linalg.norm(phys_dist)
                    # print(f'phys_dist = {phys_dist}')
                    if honeycomb:
                        if phys_dist > np.sqrt(3)/2*(j_max-1)+1:
                            continue
                    else:
                        if phys_dist > j_max:
                            continue
                    if (y-x)%3 != (jx+jy)%3:
                        continue   #condition for hole_pos and sublattices of the holes
                    
                    if jx == 0 and jy == 0:      
                        continue    #avoids putting both holes on the same site

                    too_large = False
                    if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1)) > lat_cc.depth:
                        too_large = True                

                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state, state_uncropped = add_holes(n1, n2, jx, jy, lat_sc1, lat_sc2, lat_cc, too_large=too_large)
                    # print(state, state_uncropped)
                    if type(state)==dict and not too_large:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))    #discard states that are in the cc basis, since they should be either sc+sc or cc not both
                        if found:
                            continue
                    if type(state_uncropped) == dict:
                        ### Hole 1 hops
                        for delta in [np.array([1,2]), np.array([2,1]), np.array([1,-1]), np.array([-1,-2]), np.array([-2,-1]), np.array([-1,1])]: 
                            if hole_1_hop == False:
                                continue
                            shift2 = np.ones((2,), dtype=int)*(offset-lat_sc1.depth-1)     #shift is combination of offset & lat_sc depth 
                            if np.any(np.all(sitelist2 == delta - shift2, axis=1)):  #hole1 hopping
                                continue
                            # avoid hops of hole1?! that go out of bounds
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

                            # print(f'test_lat after setting to zero: {lat2}')
                            test_lat_new = test_lat_new[1:-1, 1:-1]
                            test_hole_pos_new = [np.array(test_hole_pos_new[0]) - 1, np.array(test_hole_pos_new[1]) - 1] 
                            
                            test_state = {'lat': test_lat_new, 'hole_pos': test_hole_pos_new, 'seq': test_seq_new}
                            # print(f'test_state: {test_state}, too_large: {too_large}')
                            found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(test_state)) 
                            if found:
                                overlap_count =+1
                                repr, _, m = lat_cc.is_representative[m]
                                phys_dist = np.array(lat_cc.dist_2_phys_dist([jx, jy], seq_hole_1))
                                phys_delta = np.zeros(2)
                                phys_delta[0], phys_delta[1] = np.sqrt(3)/2*delta[0], delta[1] - 1/2*delta[0]
                                # print(f"k2 shape: {k2.shape}")
                                # print(f"phys_dist shape: {phys_dist.shape}")
                                # print(k1.shape, k2.shape, phys_dist.shape, phys_delta.shape)
                                dM0 = 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist - phys_delta) - 1j * np.einsum('ax,a->x', k1, phys_delta))
                                dM1 = 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist - phys_delta) - 1j * np.einsum('ax,a->x', k1, phys_delta))
                                # factor 1/sqrt(2) comes from projection onto ferminoic states
                                if not repr:
                                    dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist - phys_delta))
                                    dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist - phys_delta))
                                Ms0 = Ms0 + dM0
                                Ms1 = Ms1 + dM1

                        ### Hole 2 hops
                        for delta in [np.array([1,2]), np.array([2,1]), np.array([1,-1]), np.array([-1,-2]), np.array([-2,-1]), np.array([-1,1])]: 
                            if hole_2_hop == False:
                                continue
                        # for delta in [ np.array([-2,-1])]: 
                            test_hole_pos = state_uncropped['hole_pos'].copy()
                            test_lat = state_uncropped['lat'].copy()
                            lat_cc.triangular_Neel(D=lat_cc.depth+1)
                            sublattice = lat_cc.Neel_state_L_size[x]
                            # avoid hopping into strings
                            if np.any(np.all(sitelist1 - (lat_sc1.depth+1) == delta + offset , axis=1)):  #hole2 hopping
                                continue
                            # avoid hops that go out of bounds
                            if np.amax(np.abs(test_hole_pos[1]+delta-lat_cc.depth-2)) > lat_cc.depth:
                                continue
                            # print(f'offset: {offset}')
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
                                overlap_count =+1
                                
                                repr, _, m = lat_cc.is_representative[m]
                                phys_dist = np.array(lat_cc.dist_2_phys_dist([jx, jy], seq_hole_1))
                                phys_delta = np.zeros(2)
                                phys_delta[0], phys_delta[1] = np.sqrt(3)/2*delta[0], delta[1] - 1/2*delta[0]
                                dM0 = 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist))
                                dM1 = 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist))
                                # factor 1/sqrt(2) comes from projection onto ferminoic states
                                if not repr:
                                    dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist + phys_delta))
                                    dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist + phys_delta))
                                Ms0 = Ms0 + dM0
                                Ms1 = Ms1 + dM1
                                # print(f'offset: {offset}')
            pbar.update(1)
Ms_t2_0 = Ms0
Ms_t2_1 = Ms1 
np.save(f"../results/TRI/sc_cc_overlaps/M_t_L={L}_{system}_ccdepth_{depth_cc}_scdepth{depth_sc}_lmaxov_{l_max_sc_overlaps}_j{j}_jperp{j_perp}_t{t}_uc={unit_cell}.npy", (Ms_t2_0, Ms_t2_1))

print("1) J_perp overlaps")

transformed_basis = transform_lattice_j_perp(lat_cc)
Ms0 = np.zeros(k_shape)
Ms1 = np.zeros(k_shape)

overlap_counter_0 = 0
overlap_counter_1 = 0

total_iterations = len(lat_sc1.indices1)*len(lat_sc2.indices2)
with tqdm(total=total_iterations) as pbar:
    for n1 in lat_sc1.indices1:
        for n2 in lat_sc2.indices2:
            
            j_min = (np.ones(2,)*(-lat_cc.depth)).astype(int) #improve these bounds in the future
            j_max = (np.ones(2,)*(lat_cc.depth)).astype(int)

            x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])
            y = lat_sc2.find_hole_sublattice(lat_sc2.bin_basis[n2]['seq'])

            seq1 = lat_sc1.bin_basis[n1]['seq']
            seq_hole_1 = []
            if len(seq1) != 0:
                for i, step in enumerate(seq1):
                    seq_hole_1.append([0,step[0],step[1]]) #bring seq_hole_1 in the form [[0,x,y],...] add to dist_2_phys_dist

            for jx in range(j_min[0], j_max[0] + 1):          #why these ranges? why +1?
                for jy in range(j_min[1], j_max[1] + 1):

                    offset = np.array([jx, jy])
                    lat1 = lat_sc1.bin_basis[n1]['lat'].copy()
                    subl1 = lat_sc1.find_sublattice(lat_sc1.bin_basis[n1])
                    latsc1 = (lat1-subl1)%3 
                    latsc1[lat_sc1.depth+1,lat_sc1.depth+1] = 3 #to include hole in sitelist
                    sitelist1 = np.array(np.argwhere(latsc1).tolist()) 

                    lat2 = lat_sc2.bin_basis[n2]['lat'].copy()
                    subl2 = lat_sc2.find_sublattice(lat_sc2.bin_basis[n2])
                    latsc2 = (lat2-subl2)%3 
                    latsc2[lat_sc2.depth+1,lat_sc2.depth+1] = 3 #to include hole in sitelist
                    sitelist2 = np.array(np.argwhere(latsc2).tolist())

                    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   
                        continue  #skips all strings that would overlap, since the resulting state is not represented correctly at the moment
            
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
                        continue   #condition for hole_pos and sublattices of the holes
                    if jx == 0 and jy == 0:      
                        continue    #avoids putting both holes on the same site
                    # apply S ^(+)_i S^(+)_j on l.h.s. of expectation value, i.e.
                    # <cc|SS
                    state,_ = add_holes(n1, n2, jx, jy, lat_sc1, lat_sc2, lat_cc) 
                    if type(state)==dict:
                        found, _ = lat_cc.basis.search(lat_cc.state_2_list_entry(state))    #why look for this state in the lat_cc basis? avoid states that are in the cc basis, since they should be either sc+sc or cc not both
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
                                    dM0 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist))
                                    dM1 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist))
                                    # factor 1/sqrt(2) comes from projection onto fermionic states
                                    # factor 1/2 comes from H_J_perp
                                    if not repr:
                                        dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist))
                                        dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist))
                                    Ms0 = Ms0 + dM0
                                    Ms1 = Ms1 + dM1

                                    overlap_counter_0 += 1

                        # apply S ^(-)_i S^(-)_j on r.h.s. of expectation value, i.e.
                        # SS|(sc)^2>
                        ms = add_holes_j_perp(n1, n2, jx, jy, lat_sc1, lat_sc2, lat_cc)

                        for m in ms:
                            # print(f'm={m}, check')
                            repr, _, m = lat_cc.is_representative[m]
                            phys_dist = np.array(lat_sc1.dist_2_phys_dist([jx, jy],seq_hole_1))
                            dM0 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_0[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist))
                            dM1 = 0.5 * 1/np.sqrt(2) * np.conj(v_cc_1[m]) * vs_sc_1[:,n1] * vs_sc_2[:,n2] * np.exp(1j * np.einsum('ax,a->x', k2, phys_dist))
                            # factor 1/sqrt(2) comes from projection onto fermionic states
                            # factor 1/2 comes from H_J_perp
                            if not repr:
                                phys_dist = np.array(lat_sc1.dist_2_phys_dist([jx, jy],seq_hole_1))
                                dM0 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist))
                                dM1 *= -1 * np.exp(-1j * np.einsum('ax,a->x', k1 + k2, phys_dist))
                            Ms0 = Ms0 + dM0
                            Ms1 = Ms1 + dM1

                            overlap_counter_1 += 1
            pbar.update(1)
    Ms_j_0 = Ms0
    Ms_j_1 = Ms1

print("# of non-zero overlaps for Jperp:", overlap_counter_0, overlap_counter_1)
np.save(f"../results/TRI/sc_cc_overlaps/M_j_perp_L={L}_{system}_ccdepth_{depth_cc}_scdepth{depth_sc}_lmaxov_{l_max_sc_overlaps}_j{j}_jperp{j_perp}_t{t}_uc={unit_cell}.npy", (Ms_j_0, Ms_j_1))

print("Overlap calculations complete.")

import matplotlib.tri as tri
import matplotlib as mpl
import matplotlib.pyplot as plt
kx = k_grid[0, :]
ky = k_grid[1, :]
triang = tri.Triangulation(kx, ky)

mpl.rcParams['axes.titlesize'] = 14    # Size of subplot titles
mpl.rcParams['xtick.labelsize'] = 10   # Size of numbers on x-axis
mpl.rcParams['ytick.labelsize'] = 10   # Size of numbers on y-axis
mpl.rcParams['axes.labelsize'] = 12    # Size of x and y labels

fig, axs = plt.subplots(2,2, figsize=(8,7))
im1 = axs[0,0].tripcolor(triang, np.abs(Ms_t2_0).ravel(), shading='gouraud', cmap='coolwarm')
axs[0,0].set_title("$|M^{t'}_1|$")
axs[0,0].set_ylabel("$k_y$")
fig.colorbar(im1, ax=axs[0,0])
im2 = axs[0,1].tripcolor(triang, np.angle(Ms_t2_0).ravel(), shading='gouraud', cmap='twilight')
axs[1,0].set_ylabel("$k_y$")
axs[1,0].set_xlabel("$k_x$")
axs[1,0].set_title("phase($M^{t'}_1$)")
fig.colorbar(im2, ax=axs[1,0])
im3 = axs[1,0].tripcolor(triang, np.abs(Ms_t2_1).ravel(), shading='gouraud', cmap='coolwarm')
axs[0,1].set_title("$|M^{t'}_2|$")
fig.colorbar(im3, ax=axs[0,1])
im4 = axs[1,1].tripcolor(triang, np.angle(Ms_t2_1).ravel(), shading='gouraud', cmap='twilight')
axs[1,1].set_title("phase($M^{t'}_2$)")
axs[1,1].set_xlabel("$k_x$")
fig.colorbar(im4, ax=axs[1,1])
#set equal aspect ratio
axs[0,0].set_aspect('equal')
axs[0,1].set_aspect('equal')
axs[1,0].set_aspect('equal')
axs[1,1].set_aspect('equal')
plt.savefig(f"../results/figures/M_tprime_{system}_depth_sc={depth_sc}_depth_cc={depth_cc}_lmax_sc_overlaps={l_max_sc_overlaps}L={L}_jperp={j_perp}_j={j}_t={t}_uc={unit_cell}.pdf",
    bbox_inches="tight")

fig, axs = plt.subplots(2,2, figsize=(8,7))
im1 = axs[0,0].tripcolor(triang, np.abs(Ms_j_0).ravel(), shading='gouraud', cmap='coolwarm')
axs[0,0].set_title("$|M^{J_\\perp}_1|$")
axs[0,0].set_ylabel("$k_y$")
fig.colorbar(im1, ax=axs[0,0])
im2 = axs[1,0].tripcolor(triang, np.angle(Ms_j_0).ravel(), shading='gouraud', cmap='twilight')
axs[1,0].set_ylabel("$k_y$")
axs[1,0].set_xlabel("$k_x$")
axs[1,0].set_title("phase($M^{J_\\perp}_1$)")
fig.colorbar(im2, ax=axs[1,0])
im3 = axs[0,1].tripcolor(triang, np.abs(Ms_j_1).ravel(), shading='gouraud', cmap='coolwarm')
axs[0,1].set_title("$|M^{J_\\perp}_2|$")
fig.colorbar(im3, ax=axs[0,1])
im4 = axs[1,1].tripcolor(triang, np.angle(Ms_j_1).ravel(), shading='gouraud', cmap='twilight')
axs[1,1].set_title("phase($M^{J_\\perp}_2$)")
axs[1,1].set_xlabel("$k_x$")
fig.colorbar(im4, ax=axs[1,1])
#set equal aspect ratio
axs[0,0].set_aspect('equal')
axs[0,1].set_aspect('equal')
axs[1,0].set_aspect('equal')
axs[1,1].set_aspect('equal')
plt.savefig(f"../results/figures/M_jperp_{system}_depth_sc={depth_sc}_depth_cc={depth_cc}_lmax_sc_overlaps={l_max_sc_overlaps}L={L}_jperp={j_perp}_j={j}_t={t}_uc={unit_cell}.pdf",
    bbox_inches="tight")
print("Plotting complete.")
