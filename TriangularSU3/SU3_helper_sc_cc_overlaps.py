'''compute overlap between 2 sc-pairs and cc'''
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy.sparse import csr_matrix
from time import perf_counter
import copy
import cmath
from scipy.spatial import cKDTree
from importlib import reload

from . import SU3_2hole_triangular
reload(SU3_2hole_triangular)  
from .SU3_2hole_triangular import StringBasis as Lat_cc
from .SU3_2hole_triangular import sorted_list

from . import SU3_1hole_triangular
reload(SU3_1hole_triangular)  
from .SU3_1hole_triangular import StringBasis as Lat_sc1

from . import SU3_1hole_triangular2
reload(SU3_1hole_triangular2)  
from .SU3_1hole_triangular2 import StringBasis as Lat_sc2

def index2momentum(i, Lx, Ly=0, size=np.pi):    #used #returns momentum grid of size L, what is i?
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return (size*(2*i/(L)-1)) 

def make_triangular_grid_bz(L,grid_size=None): #used
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

def project_to_1bz(k_vec):
    """
    Projects a 2D vector k (or array of vectors) into the 1st Brillouin Zone.
    
    Parameters:
        k_vec: np.ndarray of shape (2,) or (N, 2)
    Returns:
        k_1bz: np.ndarray of same shape as k_vec inside the 1BZ
    """
    k_arr = np.atleast_2d(k_vec)

    # Define reciprocal basis vectors
    b1 = (2 * np.pi / 3) * np.array([np.sqrt(3), 1.0])
    b2 = (2 * np.pi / 3) * np.array([np.sqrt(3), -1.0])
    B = np.column_stack([b1, b2])      # 2x2 matrix transforming (c1, c2) -> (kx, ky)
    B_inv = np.linalg.inv(B)           # Matrix transforming (kx, ky) -> (c1, c2)
    
    # 1. Convert to fractional reciprocal coordinates
    c = k_arr @ B_inv.T  # Shape: (N, 2)
    
    # 2. Get initial integer guess
    n_guess = np.round(c)
    
    # 3. Test 3x3 local neighborhood of reciprocal lattice vectors
    offsets = np.array([
        [dx, dy] 
        for dx in [-1, 0, 1] 
        for dy in [-1, 0, 1]
    ])  # Shape: (9, 2)
    
    # Generate candidate lattice vectors G = (n + offset) @ B.T
    # Candidates shape: (N, 9, 2)
    n_candidates = n_guess[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    G_candidates = n_candidates @ B.T  # Shape: (N, 9, 2)
    
    # Calculate distance ||k - G|| for each candidate
    k_expanded = k_arr[:, np.newaxis, :]  # Shape: (N, 1, 2)
    diffs = k_expanded - G_candidates     # Shape: (N, 9, 2)
    dists_sq = np.sum(diffs**2, axis=-1)   # Shape: (N, 9)
    
    # Pick G that minimizes distance to k
    min_idx = np.argmin(dists_sq, axis=1) # Shape: (N,)
    
    # Select best diff (k - G_closest)
    k_1bz = diffs[np.arange(len(k_arr)), min_idx]
    
    return k_1bz.reshape(k_vec.shape)

def get_path_indices_aligned(k_grid, path_nodes, tolerance=1e-3):
    """
    Shifts path_nodes so the first node aligns with a k_grid point, 
    then finds ordered indices of k_grid points along the shifted path.
    
    Parameters:
    -----------
    k_grid : ndarray (2, N)
        The grid of k-points.
    path_nodes : list of ndarrays or ndarray (M, 2)
        List of symmetry points, e.g., [Gamma, K, Kp, Gamma].
    tolerance : float
        Distance threshold to capture points near the line.
        
    Returns:
    --------
    path_indices : ndarray
        Ordered indices of the k-grid points following the path.
    shifted_path_nodes : list of ndarrays
        The adjusted path nodes matching the grid.
    """
    path_nodes = [np.array(node, dtype=float) for node in path_nodes]
    
    # 1. Find distance vector 'd' from first path node to closest k_grid point
    first_node = path_nodes[0]
    distances_to_first = np.linalg.norm(k_grid - first_node[:, np.newaxis], axis=0)
    closest_idx = np.argmin(distances_to_first)
    
    # Shift vector: d = (closest grid point) - (first path node)
    d = k_grid[:, closest_idx] - first_node
    
    # 2. Shift all path nodes by d
    shifted_path_nodes = [node + d for node in path_nodes]
    
    # 3. Collect indices along the shifted segments
    path_indices = []

    for i in range(len(shifted_path_nodes) - 1):
        A = shifted_path_nodes[i]
        B = shifted_path_nodes[i+1]
        
        ba = B - A
        ba_norm_sq = np.dot(ba, ba)
        
        pa = k_grid - A[:, np.newaxis]
        
        t = np.dot(ba, pa) / ba_norm_sq
        
        closest = A[:, np.newaxis] + np.clip(t, 0, 1) * ba[:, np.newaxis]
        dist = np.linalg.norm(k_grid - closest, axis=0)
        
        mask = (dist < tolerance) & (t >= -1e-8) & (t <= 1 + 1e-8)
        segment_idx = np.where(mask)[0]
        
        sorted_idx = segment_idx[np.argsort(t[segment_idx])]
        
        # Deduplicate junction points
        if i > 0 and len(path_indices) > 0 and len(sorted_idx) > 0:
            if sorted_idx[0] == path_indices[-1]:
                sorted_idx = sorted_idx[1:]
        
        path_indices.extend(sorted_idx.tolist())

    return np.array(path_indices)

def rotate_momentum_and_M(k_grid, M, angle_rad=2*np.pi/3):
    """
    Rotates the momentum grid and reorders/maps the values of array M 
    corresponding to a 120 degree rotation (or any angle_rad).
    
    Parameters:
    -----------
    k_grid : np.ndarray
        Array of shape (2, N) containing (k_x, k_y) coordinates.
    M : np.ndarray
        Array of shape (N,) containing complex matrix element values.
    angle_rad : float
        Rotation angle in radians. Default is 2*pi/3 (120 degrees).
        
    Returns:
    --------
    k_grid_rotated : np.ndarray
        The physically rotated momentum grid of shape (2, N).
    M_rotated : np.ndarray
        The rotated/reindexed M array of shape (N,).
    """
    # 1. Define the 2D rotation matrix for 120 degrees
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R_120 = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    # 2. Rotate the coordinates of k_grid: R * k
    # k_grid shape is (2, N), so R @ k_grid rotates every (k_x, k_y) column
    k_rotated = R_120 @ k_grid
    
    # 3. Build a KDTree of the original grid to find nearest-neighbor indices
    tree = cKDTree(k_grid.T)  # k_grid.T has shape (N, 2)
    
    # Query nearest original points for each physically rotated point
    distances, mapped_indices = tree.query(k_rotated.T, k=1)
    
    # Optional check: Ensure the point mapping is valid within numerical tolerance
    if np.max(distances) > 1e-5:
        print(f"Warning: Maximum distance to nearest grid point is {np.max(distances):.2e}."
              " The grid might not be fully closed under 120-degree rotation.")
        
    # 4. Map the entries of M to the rotated grid positions
    M_rotated = M[mapped_indices]
    
    return k_rotated, M_rotated

def mirror_momentum_and_M(k_grid, M, mirror_index=0, tol=1e-5):
    """
    Applies one of the three C3-symmetric mirror reflections to a momentum grid
    and reorders/maps the corresponding values of array M.
    
    The symmetry group C_3v has 3 reflection axes at angles theta = 0, pi/3, 2pi/3
    relative to the k_x axis (corresponding to mirror normals at 0, 120, and 240 degrees).
    
    Parameters:
    -----------
    k_grid : np.ndarray
        Array of shape (2, N) containing (k_x, k_y) coordinates.
    M : np.ndarray
        Array of shape (N,) containing complex matrix element values.
    mirror_index : int
        0 : Reflection across k_x axis (kx, ky) -> (kx, -ky) [theta = 0]
        1 : Reflection across axis at 60 deg (pi/3)
        2 : Reflection across axis at 120 deg (2*pi/3)
    tol : float
        Tolerance for KDTree nearest-neighbor verification.
        
    Returns:
    --------
    k_mirrored : np.ndarray
        The physically reflected momentum grid of shape (2, N).
    M_mirrored : np.ndarray
        The reflected/reindexed M array of shape (N,).
    """
    # 1. Reflection angle of the mirror axis
    # mirror_index in {0, 1, 2} corresponds to theta in {0, pi/3, 2pi/3}
    theta = mirror_index * (np.pi / 3)
    
    # 2. General 2D Householder reflection matrix across axis at angle theta
    cos_2t = np.cos(2 * theta)
    sin_2t = np.sin(2 * theta)
    
    sigma = np.array([
        [ cos_2t,  sin_2t],
        [ sin_2t, -cos_2t]
    ])
    
    # 3. Apply reflection to grid: sigma @ k_grid
    k_mirrored = sigma @ k_grid
    
    # 4. Build a KDTree of the original grid to find nearest-neighbor indices
    tree = cKDTree(k_grid.T)  # k_grid.T has shape (N, 2)
    distances, mapped_indices = tree.query(k_mirrored.T, k=1)
    
    # Check grid closure under mirror symmetry
    max_dist = np.max(distances)
    if max_dist > tol:
        print(f"Warning: Maximum distance to nearest grid point is {max_dist:.2e}."
              f" The grid might not be fully closed under mirror #{mirror_index}.")
        
    # 5. Map the values of M to the new mirrored positions
    M_mirrored = M[mapped_indices]
    
    return k_mirrored, M_mirrored

def invert_momentum_and_M(k_grid, M, tol=1e-5):
    """
    Applies 2D inversion symmetry (kx, ky) -> (-kx, -ky) to a momentum grid
    and reorders/maps the corresponding values of array M.
    
    Parameters:
    -----------
    k_grid : np.ndarray
        Array of shape (2, N) containing (k_x, k_y) coordinates.
    M : np.ndarray
        Array of shape (N,) containing complex matrix element values.
    tol : float
        Tolerance for KDTree nearest-neighbor verification.
        
    Returns:
    --------
    k_inverted : np.ndarray
        The physically inverted momentum grid of shape (2, N).
    M_inverted : np.ndarray
        The inverted/reindexed M array of shape (N,).
    """
    # 1. Inversion operation: -k
    k_inverted = -k_grid
    
    # 2. Build KDTree of original grid to map indices
    tree = cKDTree(k_grid.T)  # k_grid.T has shape (N, 2)
    distances, mapped_indices = tree.query(k_inverted.T, k=1)
    
    # 3. Check grid closure under inversion
    max_dist = np.max(distances)
    if max_dist > tol:
        print(f"Warning: Maximum distance to nearest grid point is {max_dist:.2e}."
              " The grid might not be fully closed under inversion symmetry.")
        
    # 4. Map the entries of M to the inverted grid positions
    M_inverted = M[mapped_indices]
    
    return k_inverted, M_inverted

def sum_ind(i, j, Lx, Ly=0):
    if Ly==0:
        L = Lx
    else:
        assert i.shape[0] == 2
        assert j.shape[0] == 2
        L = np.array([Lx, Ly]).reshape((2,) + (len(i.shape)-1)*(1,))
    return np.mod(np.round(i+j-L/2), L).astype(int)

def find_l0_state_sc1(lat: Lat_sc1, n_max=7): #used; #lower n_max for testing
    '''
    Find the first energy level which has considerable quasiparticle weight i.e. overlap with the zero string-length state.
                #cc or sc?                                                                   cc or sc?
    ARGUMENTS:
    lat: Lat_sc: an instance of the sc lattice class, where the Hamiltonian has already been computed.
    n_max : int (optional): the highest energy level considered

    OUTPUT:
    n: int: gives the number of the first energy level correspondig to the l=0 state. (n=0 corresponds to the ground state)
    '''
    _, vs = lat.eigensys(n_max, full=True) # lat is Lat_sc, eigenstates up to n_max=7
    vs = np.abs(vs[:2, :])
    n= 0
    while n <= n_max:
        if vs[0, n] > vs[1, n] and vs[0, n] > 1e-4:
            break
        else:
            n += 1
    return n

def find_l0_state_sc2(lat: Lat_sc2, n_max=7): #used; #lower n_max for testing
    '''
    Find the first energy level which has considerable quasiparticle weight i.e. overlap with the zero string-length state.
                #cc or sc?                                                                   cc or sc?
    ARGUMENTS:
    lat: Lat_sc: an instance of the sc lattice class, where the Hamiltonian has already been computed.
    n_max : int (optional): the highest energy level considered

    OUTPUT:
    n: int: gives the number of the first energy level correspondig to the l=0 state. (n=0 corresponds to the ground state)
    '''
    _, vs = lat.eigensys(n_max, full=True) # lat is Lat_sc, eigenstates up to n_max=7
    vs = np.abs(vs[:2, :])
    n= 0
    while n <= n_max:
        if vs[0, n] > vs[1, n] and vs[0, n] > 1e-4:
            break
        else:
            n += 1
    return n

def init_lattices(l_max_sc, l_max_cc, l_max_sc_overlaps, j_perp_div_j=1., connected=True, honeycomb=False, unit_cell=0):
    """
    initialize and construct truncated basis for sc and cc.
    save indices of sc_basis, where string is smaller (or equal) than l_max_sc_overlaps
    Also compute Hamiltonian matrix.
    ------------
    ARGUMENTS:
    l_max_sc (int): maximal string length in sc truncated basis
    l_max_cc (int): maximal string length in cc truncated basis
    l_max_sc_overlaps (int): maximal string length considered in overlaps
    connected (bool): if True, we take only the connected strings into account
    -------------
    RETURNS:
    lat_sc: Lat_sc: lattice class from truncted basis for the sc
    lat_cc: Lat_cc: lattice class from truncted basis for the cc
    """
    lat_sc1 = Lat_sc1(l_max_sc, only_connected=connected, honeycomb=honeycomb,unit_cell=unit_cell)
    lat_sc2 = Lat_sc2(l_max_sc, only_connected=connected, honeycomb=honeycomb,unit_cell=unit_cell)
    #lat_sc.generate_basis()
    #lat_sc.matrix_el(1., 1., j_perp_div_j, 1.) #t=1, j_z=1, t2=1, j_perp=j_perp_div_j, why?
    lat_sc1.matrix_el() 
    lat_sc2.matrix_el() 

    #lat_sc.indices = list(np.argwhere(np.count_nonzero([x[0] for x in lat_sc.bin_basis], axis=(1,2)) <= l_max_sc_overlaps).flatten()) #makes list of indices(position in basis) of all states that have strings <= l_max_sc_overlaps
    lat_sc1.indices1 = list(np.argwhere(np.array([len(x['seq']) for x in lat_sc1.bin_basis]) <= l_max_sc_overlaps).flatten()) # linus
    lat_sc2.indices2 = list(np.argwhere(np.array([len(x['seq']) for x in lat_sc2.bin_basis]) <= l_max_sc_overlaps).flatten()) # linus

    lat_cc = Lat_cc(l_max_cc, only_connected=connected, honeycomb=honeycomb)
    lat_cc.matrix_el()

    print('initialized lattices')
    # print(f'dim(H_sc) = {lat_sc.basis.length}')
    # print(f'dim(H_cc) = {lat_cc.basis.length}')

    return lat_sc1, lat_sc2, lat_cc

def triangular_Neel_size_L(L,sl):
        
        if L%2 != 1:
            raise Exception(f'L = {L} not possible, Neel state lattice must have odd number of sites') 
        size = int((L-3)/2)
        triangular_lattice = np.zeros((L, L), dtype=int)
        for i in range(L):
            for j in range(L):
                if (i+j+sl+size+1) % 3 == 0: # +self.depth+1 so center site is same for all lattice sizes
                    triangular_lattice[i,j] = 0
                elif (i+j+sl+size+1) % 3 == 1:
                    triangular_lattice[i,j] = 1
                else:
                    triangular_lattice[i,j] = 2
        triangular_lattice[size+1,size+1] = 0
        #print(triangular_lattice)
        return triangular_lattice

def add_holes(n1, n2, jx, jy, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2, lat_cc: Lat_cc, too_large=False):
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
    # print(f'too_large: {too_large}')
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

    offset = np.array([jx, jy])    
    x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])
    y = lat_sc2.find_hole_sublattice(lat_sc2.bin_basis[n2]['seq'])

    extend = 2 if too_large else 0      # allow for strings that are longer by 1 site than lat_cc.depth

    skip_config = False  
    if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1)) > lat_cc.depth + extend: #skips all sc states that are too long
        skip_config = True  
    if np.amax(np.abs(sitelist1-lat_sc1.depth-1)) > lat_cc.depth + extend: #skips all sc states that are too long
        skip_config = True   
    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   #skips all strings that would overlap, since the resulting state is not represented correctly at the moment
        skip_config = True
    
                    
    if not skip_config:
        seq = []
        for move in lat_sc1.bin_basis[n1]['seq']:
            seq.append([0,move[0],move[1]])
        for move in lat_sc2.bin_basis[n2]['seq']:
            seq.append([1,move[0],move[1]])
        seq.append([1,jx,jy])
        state = -1
        if not too_large:
            lat0 = copy.deepcopy(lat_cc.Neel_state[x])
            for i in sitelist1: 
                lat0[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth)] = lat1[tuple(i)]
            for i in sitelist2:
                lat0[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth)] = lat2[tuple(i)]
            
            debug_lat = (lat0-lat_cc.Neel_state[x])%3
            hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), np.ones((2,), dtype=int) * (lat_cc.depth + 1) + offset]
            for xx in hole_pos:
                lat0[tuple(xx)] = 0
                debug_lat[tuple(xx)] = 3
                # print(f'hole at: {xx}')
            state = {'lat': lat0, 'hole_pos': hole_pos, 'seq': seq}

        #uncropped lattice for t' coupling
        sublattice = triangular_Neel_size_L(lat_cc.L_size + 2,x)
        lat0_uncropped = sublattice.copy()
        for i in sitelist1:
            lat0_uncropped[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth+1)] = lat1[tuple(i)]
        for i in sitelist2:
            lat0_uncropped[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth+1)] = lat2[tuple(i)]
        # print(f'sitelists: {sitelist1}, {sitelist2}, offset: {offset}')
        
        debug_lat_uncropped = (lat0_uncropped - sublattice)%3
        # print(f'lat_uncropped: {debug_lat_uncropped}')
        hole_pos_uncropped = [np.ones((2,), dtype=int) * (lat_cc.depth + 2), np.ones((2,), dtype=int) * (lat_cc.depth + 2) + offset]
        for xx in hole_pos_uncropped:
            lat0_uncropped[tuple(xx)] = 0
            debug_lat_uncropped[tuple(xx)] = 3
            # print(f'hole at: {xx}')
        state_uncropped = {'lat': lat0_uncropped, 'hole_pos': hole_pos_uncropped, 'seq': seq}
    else:
        state = -1 #not a dictionary, will get skipped  
        state_uncropped = -1      
    return state, state_uncropped

def cc_translation_size_L(L, lat, hole_pos, step, check=False):         #can only be used when step is in seq_hole_1  #Linus 1.0
        if step[0] != 0:
            print('wrong hole (translation)')
        x = step[1]
        y = step[2]

        if x > 0:
            lat[:x, :] = (lat[:x, :]-(L-3)/2)%3
        elif x < 0:
            lat[x:, :] = (lat[x:, :]+(L-3)/2)%3
        lat = np.roll(lat, -x, axis=0)

        if y > 0:
            lat[:,:y] = (lat[:,:y]-(L-3)/2)%3
        elif y < 0:
            lat[:,y:] = (lat[:,y:]+(L-3)/2)%3
        lat = np.roll(lat, -y, axis=1)

        hole_pos[0] = hole_pos[0] - step[1:]        #move 1st hole accordingly
        hole_pos[1] = hole_pos[1] - step[1:]        #move 2nd hole accordingly
        return lat, hole_pos

def add_holes_j_perp(n1, n2, jx, jy, lat_sc1: Lat_sc1, lat_sc2: Lat_sc2, lat_cc: Lat_cc): #used
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
    offset = np.array([jx, jy])    
    x = lat_sc1.find_hole_sublattice(lat_sc1.bin_basis[n1]['seq'])

    skip_config = False
    if np.amax(np.abs(sitelist2+offset-lat_sc1.depth-1)) > lat_cc.depth + 2: #skips all sc2 states that are longer even after flipping 2 spins
        skip_config = True    
    if np.amax(np.abs(sitelist1-lat_sc1.depth-1)) > lat_cc.depth + 2: #skips all sc1 states that are too long
        skip_config = True  
    if np.any([np.any(np.all(sitelist1 == row, axis=1)) for row in (sitelist2 + offset)]):   
        skip_config = True    #skips all strings that would overlap, since the resulting state is not represented correctly at the moment

    ms = []          
    if not skip_config: 
        sublattice = triangular_Neel_size_L(lat_cc.L_size + 4,x)
        lat0 = sublattice.copy()
        seq = [[0,x,0]]
        for i in sitelist1:
            lat0[tuple(np.array(i)-lat_sc1.depth+lat_cc.depth+2)] = lat1[tuple(i)]
        lat0[lat_cc.depth+3, lat_cc.depth+3] = 0     #mark hole site as 0
        for i in sitelist2:
            lat0[tuple(np.array(i)+np.array(offset)-lat_sc1.depth+lat_cc.depth+2)] = lat2[tuple(i)]
        lat0[tuple(np.array(offset)+lat_cc.depth+3)] = 0     #mark hole site as 0
        latcc = (lat0-sublattice)%3
        hole_pos = [np.ones((2,), dtype=int) * (lat_cc.depth + 1), offset + lat_cc.depth + 1]
        for xx in hole_pos:
            lat0[tuple(xx+2)] = 0      #set holes to zero
            latcc[tuple(xx+2)] = 0
        siteslist = np.argwhere(latcc).tolist()
        for site in siteslist:
            nx = [[1,0],[0,1],[1,1]]
            for nn in nx:
                if [site[0]+nn[0],site[1]+nn[1]] in siteslist:
                    lat1 = lat0.copy()
                    lat1[site[0],site[1]],lat1[site[0]+nn[0],site[1]+nn[1]]=lat1[site[0]+nn[0],site[1]+nn[1]],lat1[site[0],site[1]] #exchange both sites

                    #check if any spins are flipped outside L_size 
                    lat2 = lat1.copy()
                    lat2 = (lat2-sublattice)%3
                    lat2[2:-2, 2:-2] = 0 
                    if not np.all(lat2 == 0):
                        continue
                    
                    lat1 = lat1[2:-2, 2:-2]  #crop lattice back to cc size
                    state1 = {'lat': lat1, 'seq':seq, 'hole_pos': hole_pos}
                    list_entry = lat_cc.state_2_list_entry(state1)
                    found, m = lat_cc.basis.search(list_entry)
                    if found:
                        ms.append(m)
    return ms

def transform_lattice_j_perp(lat_cc: Lat_cc): #used
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
    transformed_basis = sorted_list(length_arr = lat_cc.L_size + 2)
    N = 0
    for i in range(lat_cc.basis.length):
        state = lat_cc.bin_basis[i]
        lat = state['lat']
        hole_pos = state['hole_pos']
        seq = state['seq']
        if len(seq) < 2:
            continue
        sublattice = lat_cc.find_sublattice(state)
        lat0 = (lat-sublattice)%3
        lat0[tuple(hole_pos[0])] = 0
        lat0[tuple(hole_pos[1])] = 0

        siteslist = (np.argwhere(lat0)).tolist()
        for site in siteslist: 
            nx = [[1,0],[0,1],[1,1]]  
            for nn in nx:
                if [site[0]+nn[0],site[1]+nn[1]] in siteslist:
                    lat1 = lat.copy()
                    lat1[site[0],site[1]],lat1[site[0]+nn[0],site[1]+nn[1]]=lat1[site[0]+nn[0],site[1]+nn[1]],lat1[site[0],site[1]] #exchange both sites
                    state1 = {'lat': lat1, 'seq':seq, 'hole_pos': hole_pos}
                    list_entry = lat_cc.state_2_list_entry(state1)
                    new = transformed_basis.add(list_entry)
                    _, n = transformed_basis.search(list_entry)
                    transformed_basis.list[n].append(i) #state n in transformed basis can have multiple original states leading to it
                    N += 1
    return transformed_basis


if __name__ == '__main__':
    print('starting __main__')
    j = 1/3
    j_perp = j
    t2 = 0
    depth_sc = 8
    l_max_sc_overlaps = 2
    depth_cc = 10
    p = -1
    L = 16
    Lx = L
    Ly = 4
    # k1 = np.array([1, 1]) * np.pi / 2 * 1
    # k1 = np.random.random((2,))
    # k2 = 0*np.array([np.pi, np.pi]) - k1

    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sc-cc-overlaps')

    # t0 = perf_counter()
    lats = init_lattices_cyl(depth_sc, depth_cc, l_max_sc_overlaps=l_max_sc_overlaps, Ly=Ly, j_perp_div_j=j_perp/j, connected=True)
    # compute_eigenvectors_all_momenta(L, j, j_perp, t2, *lats, Ly=Ly)
    # Ms_j, Ms_t = overlap_all_momenta_cylinder(j, L, Ly, t2, *lats)
    # print(f'Lattices initialized in {perf_counter() - t0:.2f} seconds')

    # x = 0
    # y = 0
    # Ms_j = Ms_j[:-1,:-1,:-1,:-1]
    # Ms_t = Ms_t[:-1,:-1,:-1,:-1]
    # interaction = np.roll(np.real(Ms_j)[::-1, ::-1, :, :], (x-Lx//2 + 1, y - Ly//2 + 1), axis=(0,1))
    # interaction = np.einsum('abab->ab', interaction).reshape((Lx, Ly))

    # plt.imshow(interaction.T)
    # plt.figure()
    # interaction = np.roll(np.real(Ms_t)[::-1, ::-1, :, :], (x-Lx//2 + 1, y - Ly//2 + 1), axis=(0,1))
    # interaction = np.einsum('abab->ab', interaction).reshape((Lx, Ly))
    # plt.imshow(interaction.T)
    # plt.show()


    # t0 = perf_counter()
    # lats = init_lattices(depth_sc, depth_cc, l_max_sc_overlaps=l_max_sc_overlaps, j_perp_div_j=j_perp/j, connected=True)
    # print(f'Lattices initialized in {perf_counter() - t0:.2f} seconds')

    # Ms_j, Ms_t = overlap_all_momenta_cylinder(j, )

    # t0 = perf_counter()
    # compute_eigensys_all_momenta_exc(L, j, j_perp, t2, 2, 3, *lats, p)
    # Ms_j, Ms_t = overlap_all_momenta_excited(j, L, t2, 2, 3, *lats, p)

    # t0 = perf_counter()
    # compute_eigenvectors(L, j, j_perp, t2, *lats)
    # print(f'coputed eigenvectors in {perf_counter() - t0:.2f} seconds')

    # k1 = np.array([1.0, 0.7])
    # k2 = np.array([0.05, 0.01])

    # x = overlap_t(k1, k2, j, t2, *lats)
    # print('-------------------------------------------------')
    # y = overlap_t(k2, k1, j, t2, *lats)
    # print(np.abs(x + y))
    # print(x, y)
    # print(np.abs(x), np.abs(y), np.angle(x/y))

    # t0 = perf_counter()
    # Ms = overlap_grid(j, L, t2, *lats)
    # print(f'Computed overlap on grid in {perf_counter() - t0:.2f} seconds')
    # # np.save(os.path.join(path, f'overlaps_{depth_sc}.npy'), Ms)
    # if np.amax(np.abs(np.imag(Ms))) > 0:
    #     print('WARNING: Overlaps are not real')
    #     Ms = np.real(Ms)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$k_x/\pi$')
    # ax.set_ylabel('$k_y/\pi$')
    # ax.set_title('overlaps, $l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    # fig.colorbar(im)

    # t0 = perf_counter()
    # Ms = overlap_grid_j(j, L, t2, *lats)
    # print(f'Computed overlap J on grid in {perf_counter() - t0:.2f} seconds')
    # # np.save(os.path.join(path, f'overlaps_j_{depth_sc}.npy'), Ms)
    # if np.amax(np.abs(np.imag(Ms))) > 0:
    #     print('WARNING: Overlaps J are not real')
    #     Ms = np.real(Ms)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$k_x/\pi$')
    # ax.set_ylabel('$k_y/\pi$')
    # ax.set_title('$M_{J}(k), \; l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    # fig.colorbar(im)
    # # test_symmetries(Ms)

    # t0 = perf_counter()
    # Ms = overlap_grid_t(j, L, t2, *lats)
    # print(f'Computed overlap t\' on grid in {perf_counter() - t0:.2f} seconds')
    # # np.save(os.path.join(path, f'overlaps_t_{depth_sc}.npy'), Ms)
    # if np.amax(np.abs(np.imag(Ms))) > 0:
    #     print('WARNING: Overlaps t2 are not real')
    #     Ms = np.real(Ms)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$k_x/\pi$')
    # ax.set_ylabel('$k_y/\pi$')
    # ax.set_title('$M_{t\'}(k),\; l_{max}=$'+f'{depth_sc}')
    # im = ax.imshow(Ms.T, origin='lower', extent=[-1,1,-1,1], cmap='RdBu')
    # fig.colorbar(im)
    # # test_symmetries(Ms)

    # print(Ms[L//2, -1])

    # plt.show()

    # ----------------------------------------------------------------------------

    # for l in range(5, 6):
    #     load_plots(l, '_neg')
    # # plt.show()

    # # j = 1/3
    # # t2 = 0.2
    # # load_total(5, 'pos', j, t2)
    # # load_total(5, 'neg', j, t2)
    # # plt.show()

    # # convergence_plots(4, 1)
    # plt.show()

    # ---------------- begin test ----------------
    # test_symmetries(parity=+1)
    # ---------------- end test ------------------


    ### compute eigenenergies
    jperps = [0.1*j, 0.5*j, j]
    for j_perp in jperps:
        compute_eigenenergies_all_momenta(L, j, j_perp, t2, *lats, p)
        # compute_eigenenergies_all_momenta(L, j, j_perp, t2, *lats, -p)

    # # --------------------------------------
    # k = np.array([-3/8, 0])*np.pi
    # load_plots_all_momenta(k, j=1/3, t2=0., depth_overlap=3, p=1)
    # load_plots_all_momenta(k, j=0, t2=-0.2, depth_overlap=3, p=1)
    # k = np.array([-1, 0])*np.pi
    # load_plots_all_momenta(k, j=1/3, t2=0., depth_overlap=3, p=1)
    # load_plots_all_momenta(k, j=0, t2=-0.2, depth_overlap=3, p=1)
    # plt.show()