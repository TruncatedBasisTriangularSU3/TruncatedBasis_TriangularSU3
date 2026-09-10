'''compute overlap between 2 sc-pairs and cc'''
import numpy as np
import matplotlib.pyplot as plt
import os.path

from .HC_1_hole import StringBasisHC as basis_sc
from .HC_2_holes import sorted_list
from .HC_2_holes import StringBasis as basis_cc

def plot_lat(state, ax=None, end=False, fig=None, axs=None):
    if isinstance(state, dict):
        lat = state['lat'].astype(float)
        if 'hole_pos' in state.keys():
            lat[state['hole_pos'][0][0], state['hole_pos'][0][1]] = 0.5
            lat[state['hole_pos'][1][0], state['hole_pos'][1][1]] = 0.5
        else: 
            lat[lat.shape[0]//2, lat.shape[1]//2] = 0.5
    else:
        lat = state.astype(float)
    if not ax is None:
        axs[ax].imshow(lat, vmin=0, vmax=1, cmap='RdPu')
        if end:
            plt.show()
    else:
        plt.imshow(lat, vmin=0, vmax=1, cmap='RdPu')
        plt.show()

def index2momentum(i, Lx, Ly=0, size=np.pi):    #returns momentum grid of size L, what is i?
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return (size*(2*i/(L)-1)) 

def make_triangular_grid_bz(L,grid_size=None): 
    'input L: odd integer'
    'output k_grid: array of shape (2, N) with N=1+3/4(L^2-1) k-points on hexagonal grid'
    size = 2*np.pi/np.sqrt(3) * L/(L-1)
    if grid_size != None:
        size = grid_size

    g = np.array([[1/3, -1/3],[1/np.sqrt(3), 1/np.sqrt(3)]])

    mom_ind = np.indices((L, L))
    k_grid = index2momentum(mom_ind, L, L, size=size)
    k_grid = k_grid + size/L
    rows, cols = np.indices(k_grid[0].shape) 
    mask1 = (rows + cols) >= L/2 - 1
    mask2 = (rows + cols) <= 3/2*L - 1 
    mask = mask1 & mask2

    k_x = k_grid[0][mask]
    k_y = k_grid[1][mask]
    k_grid = np.stack([k_x, k_y])
    k_grid = np.einsum('ij, jk -> ik', g, k_grid)
    return k_grid

def sum_ind(i, j, Lx, Ly=0):
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        assert j.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return np.mod(np.round(i+j-L/2), L).astype(int)

def brick_to_hc_distance(d):
        ''' converts distance dx, dy on brickwall lattice to distance r1, r2 on honeycomb lattice '''
        assert np.sum(d)%2 == 0 #check that d is already resized to same sl distance
        r1 = -d[0]
        r2 = (d[1]+d[0])//2
        return r1, r2

def find_l0_state(lat: basis_sc, n_max=7):
    '''
    Find the first energy level which has considerable quasiparticle weight i.e. overlap with the zero string-length state.

    ARGUMENTS:
    lat: Lat_sc: an instance of the sc lattice class, where the Hamiltonian has already been computed.
    n_max : int (optional): the highest energy level considered

    OUTPUT:
    n: int: gives the number of the first energy level correspondig to the l=0 state. (n=0 corresponds to the ground state)
    '''
    _, vs = lat.eigensys(n_max, full=True)
    vs = np.abs(vs[:2, :])
    n = 0
    while n <= n_max:
        if vs[0, n] > vs[1, n] and vs[0, n] > 1e-4:
            break
        else:
            n += 1
    return n

def init_lattices(l_max_sc, l_max_cc, l_max_sc_overlaps, j_perp_div_j=1., connected=True):
    """
    initialize and construct truncated basis for sc and cc.
    save indices of sc_basis, where string is smaller (or equal) than l_max_sc_overlaps
    Also compute Hamiltonian matrix.
    ------------
    ARGUMENTS:
    l_max_sc (int): maximal string length in sc truncated basis
    l_max_cc (int): maximal string length in cc truncated basis
    l_max_sc_overlaps (int): maximal string lemgth considered in overlaps
    connected (bool): if True, we take only the connected strings into account
    -------------
    RETURNS:
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    """
    lat_sc_1 = basis_sc(l_max_sc, only_connected=connected, initial_sl=0)
    lat_sc_2 = basis_sc(l_max_sc, only_connected=connected, initial_sl=1)

    lat_sc_1.indices = list(np.argwhere(np.count_nonzero([x[0] for x in lat_sc_1.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten())
    lat_sc_1.indices = list(np.argwhere(np.count_nonzero([x[0] for x in lat_sc_2.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten())

    lat_cc = basis_cc(l_max_cc, only_connected=connected)

    print('initialized lattices')
    print(f'dim(H_sc_1) = {lat_sc_1.basis.length}')
    print(f'dim(H_sc_2) = {lat_sc_2.basis.length}')
    print(f'dim(H_cc) = {lat_cc.basis.length}')

    return lat_sc_1, lat_sc_2, lat_cc
    

def add_holes_hc(state1, state2, hole_pos_2, lat_sc_1, lat_sc_2, lat_cc):
    '''
    Takes 2 string states for the magnetic polaron (sc) with distance hole_pos_2 and create dictionary state in the format of Lat_cc
    ------------
    ARGUMENTS:
    n1 (int): index of string state 1 in sc truncated basis
    n2 (int): index of string state 2 in sc truncated basis
    hole_pos_2 (array of shape (2,)): gives position of the additional hole we poke into the system in the LLP frame of the first hole
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    state: dictionary with lat, hole_pos, seq: state of the 2 sc's in the format of Lat_cc
    '''
    j_max = np.amax(np.abs(hole_pos_2))
    jy = hole_pos_2[0]
    jx = hole_pos_2[1]
    sl_0 = state1['sl']
    sl_1 = state2['sl']
    lat1 = state1['lat']
    lat2 = state2['lat']
    
    lat1 = np.pad(lat1, (j_max, j_max), 'constant', constant_values=False) #pads such that hole 1 sits at center
    lat2 = np.pad(lat2, ((j_max + jy, j_max - jy), (j_max + jx, j_max - jx)), 'constant', constant_values=False) #pads such that hole 2 sites at jx,jy
    
    lat = np.logical_xor(lat1, lat2)
    # now crop lat so that it has the same dimension as lattices in Lat_cc
    dx = lat_sc_1.depth + j_max - lat_cc.depth
    if dx > 0: #new lattice is too big, need to crop
        count = np.count_nonzero(lat[0:dx,:]) + np.count_nonzero(lat[-dx:,:]) + np.count_nonzero(lat[:,0:dx]) + np.count_nonzero(lat[:,-dx:]) #count strings outside cropped region
        if count > 0: #if strings outside, state not possible
            return -1
        else:
            lat = lat[dx:-dx, dx:-dx]
    elif dx < 0:
        lat = np.pad(lat, (-dx, -dx), 'constant', constant_values=False)
    hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + hole_pos_2] #holes at center and at hole_pos_2
    for x in hole_pos:
        lat[tuple(x)] = False #set hole positions to False (need to add sl here)
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    return state

def add_holes_hc_hopping(state1, state2, hole_pos_2, lat_sc_1, lat_sc_2, lat_cc):
    '''
    Takes 2 string states for the magnetic polaron (sc) with distance hole_pos_2 and create dictionary state in the format of Lat_cc
    ------------
    ARGUMENTS:
    n1 (int): index of string state 1 in sc truncated basis
    n2 (int): index of string state 2 in sc truncated basis
    hole_pos_2 (array of shape (2,)): gives position of the additional hole we poke into the system in the LLP frame of the first hole
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    state: dictionary with lat, hole_pos, seq: state of the 2 sc's in the format of Lat_cc
    '''
    j_max = lat_cc.depth + 2
    jy = hole_pos_2[0]
    jx = hole_pos_2[1]
    sl_0 = state1['sl']
    sl_1 = state2['sl']
    lat1 = state1['lat']
    lat2 = state2['lat']
    #print(j_max)
    #print("lat1, lat2:", lat1.shape, lat2.shape)
    lat1 = np.pad(lat1, (j_max, j_max), 'constant', constant_values=False) #pads such that hole 1 sits at center
    lat2 = np.pad(lat2, ((j_max +jy, j_max - jy), (j_max + jx, j_max - jx)), 'constant', constant_values=False) #pads such that hole 2 sites at jx,jy
    #print("lat1, lat2:", lat1.shape, lat2.shape)
    lat = np.logical_xor(lat1, lat2)
    #print("lat:", lat.shape)
    if np.any(np.abs(np.array([jx,jy]))>lat_cc.depth+1):
        state_phys = -1 #hole placed outside cc lattice
    else:
        # now crop lat so that it has the same dimension as lattices in Lat_cc
        dx = lat_sc_1.depth + 2 #j_max - (lat_cc.depth)
        if dx >= 0: #new lattice is too big, need to crop
            count = np.count_nonzero(lat[0:dx,:]) + np.count_nonzero(lat[-dx:,:]) + np.count_nonzero(lat[:,0:dx]) + np.count_nonzero(lat[:,-dx:]) #count strings outside cropped region
            if count > 0: #if strings outside, state not possible
                state_phys = -1
            else:
                lat_phys = lat[dx:-dx, dx:-dx]
                hole_pos_phys = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + hole_pos_2] #holes at center and at hole_pos_2
                for x in hole_pos_phys:
                    lat_phys[tuple(x)] = False #set hole positions to False (need to add sl here)
                state_phys = {'lat': lat_phys, 'hole_pos': hole_pos_phys, 'sl': [sl_0, sl_1]}
                
        elif dx < 0:
            lat_phys = np.pad(lat, (-dx, -dx), 'constant', constant_values=False)
            hole_pos_phys = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + hole_pos_2] #holes at center and at hole_pos_2
            for x in hole_pos_phys:
                lat_phys[tuple(x)] = False #set hole positions to False (need to add sl here)
            state_phys = {'lat': lat_phys, 'hole_pos': hole_pos_phys, 'sl': [sl_0, sl_1]}
    
    #state to work with is bigger
    # if lat.shape[0]!= (2*(lat_cc.depth+3)+1):
    #     print(lat.shape, (2*(lat_cc.depth+3)+1))
    #     pad = 2*lat_cc.depth+7-lat.shape[0]
    #     lat = np.pad(lat, (pad//2, pad//2), 'constant', constant_values=False)
    large_depth = lat_sc_1.depth + j_max
    hole_pos = [np.ones((2,), dtype=int) * (large_depth +1), np.ones((2,), dtype=int) * (large_depth+1) + hole_pos_2] #holes at center and at hole_pos_2
    for x in hole_pos:
        lat[tuple(x)] = False #set hole positions to False (need to add sl here)
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    
    return state, state_phys #this is now one site larger than lat_cc states at each axis!


def crop_lattice(lat, depth_sc, depth_cc, hole_pos_2, sl):
    '''
    Takes a lattice of spin configurations from two sc's, crops it and gives out a state dictionary
    ------------
    ARGUMENTS:
    lat: array of shape(N, N) of bool: spin configuration of (sc)^2
    depth_sc (int): l_max for lat_sc
    depth_cc (int): l_max for lat_cc
    hole_pos_w: arrow of shape(2,) of int: posotion of second hole
    -------------
    RETURNS:
    state: dict: state as in Lat_cc
    '''
    seq = []
    j_max = np.amax(np.abs(hole_pos_2))
    dx = depth_sc + j_max - depth_cc
    if dx > 0:
        count = np.count_nonzero(lat[0:dx,:]) + np.count_nonzero(lat[-dx:,:]) + np.count_nonzero(lat[:,0:dx]) + np.count_nonzero(lat[:,-dx:])
        if count > 0:
            return -1
        else:
            lat = lat[dx:-dx, dx:-dx]
    elif dx < 0:
        lat = np.pad(lat, (-dx, -dx), 'constant', constant_values=False)
    hole_pos = [np.ones((2,), dtype=int) * (depth_cc + 1), np.ones((2,), dtype=int) * (depth_cc + 1) + hole_pos_2]
    for x in hole_pos:
        lat[tuple(x)] = False
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': sl}
    return state

def add_holes_j_perp(state1, state2, hole_pos_2, lat_sc_1, lat_sc_2, lat_cc, plot=False):
    '''
    Takes 2 string states for the magnetic polaron (sc) with distance hole_pos_2, apply H_J_perp^(-) (i.e. only flipping frustrated (wrt Neel) spins)
    and create dictionary state in the format of Lat_cc
    ------------
    ARGUMENTS:
    n1 (int): index of string state 1 in sc truncated basis
    n2 (int): index of string state 2 in sc truncated basis
    hole_pos_2 (array of shape (2,)): gives position of the additional hole we poke into the system in the LLP frame of the first hole
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    ms: list of int: indices of cc states 
    '''
    j_max = np.amax(np.abs(hole_pos_2))
    jy = hole_pos_2[0]
    jx = hole_pos_2[1]
    lat1 = state1['lat']
    lat2 = state2['lat']
    sl_0 = state1['sl']
    sl_1 = state2['sl']
    lat1 = np.pad(lat1, (j_max, j_max), 'constant', constant_values=False)
    lat2 = np.pad(lat2, ((j_max + jy, j_max - jy), (j_max + jx, j_max - jx)), 'constant', constant_values=False)
    lat = np.logical_xor(lat1, lat2)


    hole_pos = [np.ones((2,), dtype=int) * (j_max + lat_sc_1.depth + 1), np.ones((2,), dtype=int) * (j_max + lat_sc_1.depth + 1) + hole_pos_2]
    for x in hole_pos:
        lat[tuple(x)] = False
    state = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    state_prev = {'lat': lat, 'hole_pos': hole_pos, 'sl': [sl_0, sl_1]}
    ms = []

    siteslist = list(np.argwhere(lat))
    for site in siteslist:
        lat_temp = lat.copy()
        if (np.sum(site-hole_pos[0])+sl_0)%2==1: #bond upwards
            if lat_temp[site[0], site[1]] and lat_temp[site[0]+1, site[1]]:
                lat_temp[site[0], site[1]] = False
                lat_temp[site[0]+1 ,site[1]] = False
                state = crop_lattice(lat_temp, lat_sc_1.depth, lat_cc.depth, hole_pos_2, [sl_0, sl_1])
                if type(state)==dict:
                    found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                    if found:
                        if plot:
                            fig, axs = plt.subplots(1,4, figsize=(12,4))
                            plot_lat(state1, ax=0, axs=axs, fig=fig)
                            plot_lat(state2, ax=1, axs=axs, fig=fig)
                            plot_lat(state_prev, ax=2, axs=axs, fig=fig)
                            plot_lat(state, ax=3, axs=axs, fig=fig, end=True)
                        ms.append(m)

        lat_temp = lat.copy()
        if lat_temp[site[0], site[1]] and lat_temp[site[0], site[1]+1]:
            lat_temp[site[0], site[1]] = False
            lat_temp[site[0], site[1]+1] = False
            state = crop_lattice(lat_temp, lat_sc_1.depth, lat_cc.depth, hole_pos_2, [sl_0, sl_1])
            if type(state)==dict:
                #plot_lat(state)
                found, m = lat_cc.basis.search(lat_cc.state_2_list_entry(state))
                if found:
                    if plot:
                        fig, axs = plt.subplots(1,4, figsize=(12,4))
                        plot_lat(state1, ax=0, axs=axs, fig=fig)
                        plot_lat(state2, ax=1, axs=axs, fig=fig)
                        plot_lat(state_prev, ax=2, axs=axs, fig=fig)
                        plot_lat(state, ax=3, axs=axs, fig=fig, end=True)
                    #print(m)
                    ms.append(m)
    return ms

def transform_hc_lattice_j_perp(lat_cc):
    '''
    Builds a basis with all possible states of the form H_J_perp|n_cc>
    ------------
    ARGUMENTS:
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    -------------
    RETURNS:
    transformed_basis: sorted_list: sorted_list of all transformed basis states
                        entries are lists with [:L_size] = lat
                        [L_size, L_size+2] = hole_pos
                        [L_size+2, :] originial indices, i.e. index in lat_cc to know from which string state the n-th transformed state was obtained
    '''
    transformed_basis = sorted_list(length_arr = lat_cc.L_size + 3)

    for i in range(lat_cc.basis.length):
        state = lat_cc.bin_basis[i]
        lat = state['lat']
        hole_pos = state['hole_pos']
        sl = state['sl']

        siteslist = list(np.argwhere(lat))

        for site in siteslist:
            lat1 = lat.copy()
            if (lat_cc.L_size*site[0]+site[1]+sl[0])%2==1: #bond downwards
                if lat1[site[0], site[1]] and lat1[site[0]+1, site[1]]: #flip along y
                    lat1[site[0], site[1]] = False
                    lat1[site[0]+1, site[1]] = False
                    a = lat1*np.matmul(np.ones((lat_cc.L_size, 1), dtype=np.uint32),
                        np.reshape(2**np.arange(lat_cc.L_size, dtype=np.uint32), (1, lat_cc.L_size)), dtype=np.uint32)
                    _ = transformed_basis.add((np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos] +[sl]))
                    _, n = transformed_basis.search(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos] +[sl])
                    transformed_basis.list[n].append(i)
                    
                    lat1[site[0], site[1]] = True
                    lat1[site[0]+1, site[1]] = True
            
            if lat1[site[0], site[1]] and lat1[site[0],site[1]+1]:
                lat1[site[0], site[1]] = False
                lat1[site[0], site[1]+1] = False
                new_state = {'lat': lat1, 'hole_pos': hole_pos, 'sl': sl}
                a = lat1*np.matmul(np.ones((lat_cc.L_size, 1), dtype=np.uint32), np.reshape(2**np.arange(lat_cc.L_size, dtype=np.uint32), (1, lat_cc.L_size)), dtype=np.uint32)
                _ = transformed_basis.add(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos]+[sl])
                _, n = transformed_basis.search(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos]+[sl])
                transformed_basis.list[n].append(i)
    

    return transformed_basis

