 # import necessary libraries
import numpy as np
import copy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from time import perf_counter
import matplotlib.pyplot as plt
import os

def greater_arr(a, b, length):
### lexicographic comparison of lists
    return a[:length]>b[:length]

class sorted_list:
### defines a sorted list without dublicates allowing a fast search of elements
    def __init__(self,length_arr):
        self.length = 0
        self.list = []
        self.length_arr=length_arr

    def add(self, a):
        # adds the element a to the right position in the list
        imin = 0
        imax = self.length-1
        while imin <= imax:
            i = (imax-imin)//2+imin
            if greater_arr(a, self.list[i],self.length_arr):
                imin = i+1
            elif greater_arr(self.list[i], a,self.length_arr):
                imax = i-1
            else:  # a is already in the list
                imax = -2
        new = False
        if imax != -2:
            self.list.insert(imin, a)
            self.length += 1
            new = True
        return new
    
    def search(self,a):
    # method to check if an element is already included in the list
    # returns a (found, i) where found = bool indicating if a is in the list
    #                            i = int gives the position of a in the list (if found == True)
        found = False
        imin = 0
        imax = self.length-1
        i = -1
        while imin <= imax and not(found):
            i = (imax-imin)//2+imin
            if greater_arr(a, self.list[i],self.length_arr):
                imin = i+1
            elif greater_arr(self.list[i], a,self.length_arr):
                imax = i-1
            else:
                found = True  
        return found, i

class StringBasis:
# A class for generating a truncated basis
    def __init__(self, depth, only_connected=True, honeycomb=False, unit_cell=0):  # works only for depth<=14 otherwise change uint32 to uint64!
        self.depth = depth
        self.L_size = 2*self.depth+3
        self.basis = sorted_list(length_arr = self.L_size + 2)
        self.bin_basis=[]
        self.last_move = [None, None]
        self.moves = [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 1, 1], [0, -1, -1]] + [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [1, 1, 1], [1, -1, -1]] #all moves for first hole and for second hole
        self.initial_moves = [[0, 1, 0], [0, 0, 1], [0, -1, -1]] + [[1, 1, 0], [1, 0, 1], [1, -1, -1]]  #for building basis, using charge conservation (no need for the other 3 depth = 1 basis states)
        self.H=None
        self.col_t=[]   #matrix elemnt now saved here not in matrix_el() function, why?
        self.row_t=[]
        self.data_t=None
        self.col_t2 = []
        self.row_t2 = []
        self.data_t2 = None
        self.col_j=[]
        self.row_j=[]
        self.data_j=[]
        self.representatives = []
        self.is_representative = []
        self.only_connected = only_connected
        self.honeycomb = honeycomb
        self.unit_cell = unit_cell
        if unit_cell is None:
            self.big_unit_cell = False
        elif isinstance(unit_cell, int):
            self.big_unit_cell = True
        print(f'2hole big_unit_cell={self.big_unit_cell}, unit_cell={unit_cell}')
        
        
        self.tol=0
        self.Neel_state_L_size = []
        self.Neel_state = []
        self.triangular_Neel()
        self.generate_basis()
        self.order_basis()
        self.generate_representatives()         #introducing representatives to make the holes distinguishable
        self.matrix_el()

        self.data_t_test = []
        self.data = []

    def triangular_Neel(self, D=0):      #linus 1.0
        L_size = self.L_size
        depth = self.depth
        if D != 0:
            depth = D
            L_size = 2*D+3
        # generates the three possible neel states (center site = 0,1,2 i.e red, green, blue) and saves them to list, because they are needed often
        for k in range(3):
            triangular_lattice = np.zeros((L_size, L_size), dtype=int)
            for i in range(L_size):
                for j in range(L_size):
                    if (i+j+k+depth+1) % 3 == 0: # +self.depth+1 so center site is same for all lattice sizes
                        triangular_lattice[i,j] = 0
                    elif (i+j+k+depth+1) % 3 == 1:
                        triangular_lattice[i,j] = 1
                    else:
                        triangular_lattice[i,j] = 2
            triangular_lattice[depth+1,depth+1] = 0
            #print(triangular_lattice)
            if D==0:
                self.Neel_state.append(triangular_lattice)
            else:
                self.Neel_state_L_size.append(triangular_lattice)

    def find_sublattice(self,state):      #linus 1.0
        seq = state['seq']
        #select part of sequence that moves 1st hole
        seq_hole_1 = []
        for move in seq:
            if move[0] == 0:
                seq_hole_1.append(move)
        y = np.sum(np.array(seq_hole_1), axis=0, dtype=int)
        y = (np.sum(y))%3
        sublattice = self.Neel_state[y]         #unprotected by deepcopy
        return sublattice 

    def find_hole_sublattice(self,seq):      #linus 1.0
        seq_hole_1 = []
        seq_hole_2 = []
        for move in seq:
            if move[0] == 0:
                seq_hole_1.append(move[1:])
            else:
                seq_hole_2.append(move[1:])
        
        x = np.sum(np.array(seq_hole_1), axis=0, dtype=int)
        x = (np.sum(x))%3
        y = np.sum(np.array(seq_hole_2), axis=0, dtype=int)
        y = (np.sum(y))%3
        hole1_sublattice = x
        hole2_sublattice = y
        return hole1_sublattice, hole2_sublattice   
    
    def hole_is_on_2_sublattice(self,seq):
        innit = False
        seq_hole_1 = []
        seq_hole_2 = []
        for move in seq:
            if move[0] == 0: 
                seq_hole_1.append(move[1:])
            else:
                seq_hole_2.append(move[1:])
        
        x = np.sum(np.array(seq_hole_1), axis=0, dtype=int)
        x = (np.sum(x))%3
        y = np.sum(np.array(seq_hole_2), axis=0, dtype=int)
        y = (np.sum(y))%3
        if x == 2 or y == 2:
            innit = True
        
        return innit
    
    def dist_2_uc_dist0(self,dist,seq):   # horizontal unit cell

        dist1 = dist.copy()
        if self.honeycomb:
            x = np.sum(np.array(dist1))
            if x % 3 == 1:
                dist1[1] += -1
            elif x % 3 == 2:
                dist1[1] += 1
        elif not self.honeycomb:
            z,_ = self.find_hole_sublattice(seq)
            z = (-z+1)%3
            x,y = dist1[0]%3, dist1[1]%3
            dist1[1] += (z-x-y)%3-z
        return dist1
    
    def dist_2_uc_dist1(self,dist,seq):   # vertical unit cell, only honeycomb uc adjusted
        dist1 = dist.copy()
        if self.honeycomb:
            x = np.sum(np.array(dist1))
            if x % 3 == 1:
                dist1[0] += -1
            elif x % 3 == 2:
                dist1[0] += 1
        elif not self.honeycomb:
            z,_ = self.find_hole_sublattice(seq)
            z = (-z+1)%3
            x,y = dist1[0]%3, dist1[1]%3
            dist1[1] += (z-x-y)%3-z
        return dist1
    
    def dist_2_uc_dist2(self,dist,seq):  # diagonal unit cell, only honeycomb uc adjusted
        dist1 = dist.copy()
        if self.honeycomb:
            x = np.sum(np.array(dist1))
            if x % 3 == 1:
                dist1[0] += 1
                dist1[1] += 1
            elif x % 3 == 2:
                dist1[0] -= 1
                dist1[1] -= 1
        elif not self.honeycomb:
            z,_ = self.find_hole_sublattice(seq)
            z = (-z+1)%3
            x,y = dist1[0]%3, dist1[1]%3
            dist1[1] += (z-x-y)%3-z
        return dist1
    
    def dist_2_phys_dist(self,dist,seq):   #transforms square lattice with one diagonal coupling to triangular lattice (i,j) = (sqrt(3)/2*i, j - 1/2*i)
        # to acces all three gauges: C3 rotate dist, apply dist_2_uc_dist0, rotate back
        phys_dist = np.zeros(2)
        if self.big_unit_cell:
            if self.unit_cell == 0:
                dist = self.dist_2_uc_dist0(dist,seq)
            elif self.unit_cell == 1:
                dist = [-dist[1],dist[0]-dist[1]]
                dist = self.dist_2_uc_dist0(dist,seq)
                dist = [dist[1]-dist[0],-dist[0]]
            elif self.unit_cell == 2:
                dist = [dist[1]-dist[0],-dist[0]]
                dist = self.dist_2_uc_dist0(dist,seq)
                dist = [-dist[1],dist[0]-dist[1]]
        phys_dist[0], phys_dist[1] = np.sqrt(3)/2*dist[0], dist[1] - 1/2*dist[0]
        return np.array(phys_dist)
    
    def connected(self, state):                 #linus 1.0
        res = False
        lat = state['lat']
        hole_pos = state['hole_pos']

        sublattice = self.find_sublattice(state)
        lat = (lat-sublattice)%3 #subtract background
        lat[self.depth+1,self.depth+1] = 0
        # construct list of sites with flipped spins
        flipped = (np.argwhere(lat)).tolist()
        flipped = flipped + [list(hole_pos[0])] + [list(hole_pos[1])]   #includes both hole positions to list
        # determine if the flipped spins are connected
        if len(flipped) == 0:
            res = True
        else:
            moves = list(np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1],[-1, -1]]))
            comp = [flipped.pop()]
            n = 0
            while len(flipped) > 0 and n < len(comp):
                r = comp[n]
                n += 1
                for move in moves:
                    testsite = list(np.array(r) + move)
                    if testsite in flipped:
                        comp.append(flipped.pop(flipped.index(testsite)))
            res = len(flipped) == 0
            # if not res:
            #     print(state, 'is not connected')
        return res

    def state_2_list_entry(self, state):        #linus 1.0
        lat = state['lat']
        hole_pos = state['hole_pos'] 
        sublattice = self.find_sublattice(state)
        # print(f'subl shape:{np.shape(sublattice)}')
        # print(f'lat shape:{np.shape(lat)}')
        lat = (lat-sublattice)%3

        
        x = hole_pos[0] 
        y = hole_pos[1] 
        lat[tuple(x)] = 0   #mark both hole sites as zero
        lat[tuple(y)] = 0   # are depth=1 states indistinguishable? -> no, since hole pos are encoded in both basis reps

        # print('lat in state_2_listentry',lat)

        if any(y < 0) or any(y >= self.L_size):
            print()
            print(f'hole out of bounds with hole_pos:{hole_pos}')
            print(f'lat:{lat}') 
            raise Exception('hole out of bounds')
        
        a = lat*np.matmul(np.ones((self.L_size, 1), dtype=np.uint64),
                          np.reshape(3**np.arange(self.L_size, dtype=np.uint64), (1, self.L_size)), dtype=np.uint32)
        list_entry = np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos]
        return list_entry 

    def generate_step(self, l, state0): #not used: creates one branch after thus not saving shortest sequence for state that can be generated by multible sequences -> less efficient than generate_basis
    # function which recursively generates all possible paths
        for move in self.moves:
            state_initial = copy.deepcopy(state0)
            physical, state = self.generate_basis_element(state_initial, np.array(move, dtype=int))
            if l > 1 and physical:
                self.generate_step(l-1, state)
           
    def generate_basis_element(self, state, step):         #linus 1.0
    # an old state and a step this function generetates a basis element and adds it to the basis list
    # state = dict(')
    # seq is a list of steps = np.arrays where seq[i] = [n, x, y]
    # n = 0,1 denotes if the first (0) or second (1) hole moves
    # x,y in {-1, 0, 1} denote the direction of the hopping
        lat, hole_pos = self.make_step(state['lat'], state['hole_pos'], step)
        seq = state['seq'] + [step]
        state = {'lat': lat, 'hole_pos': hole_pos, 'seq': seq}

        # check if the two holes site on the same site
        assert (hole_pos[0]==self.depth + 1).all()      #returns error if first hole is not in center
        if self.only_connected:
            physical = (not (hole_pos[0] == hole_pos[1]).all()) and self.connected(state)   #2 holes must be on different sites
        else:
            physical = not (hole_pos[0] == hole_pos[1]).all()   #2 holes must be on different sites
        new = False
        if physical:
            new = self.basis.add(self.state_2_list_entry(state) + [self.basis.length])

        check = new and physical
        if check:
            self.bin_basis.append(state)
    
    def translation(self, lat, hole_pos, step, D=0, debug=False):         #can only be used when step is in seq_hole_1  #Linus 1.0
        L_size = self.L_size
        depth = self.depth
        if D != 0:
            depth = D
            L_size = 2*D+3

        if step[0] != 0:
            print('wrong hole (translation)')
        x = step[1]
        y = step[2]
        if any(hole_pos[1]-step[1:] < 0) or any(hole_pos[1]-step[1:] >= L_size):
            print(f'step={step} out of bounds')
            print(f'with lat:{lat}, hole_pos:{hole_pos}') 
            raise Exception('out of bounds')
        if debug:
            print('lat pre translation',lat)
        if x > 0:
            lat[:x, :] = (lat[:x, :]-depth)%3
        elif x < 0:
            lat[x:, :] = (lat[x:, :]+depth)%3
        lat = np.roll(lat, -x, axis=0)
        if debug:
            print('lat post x translation',lat)

        if y > 0:
            lat[:,:y] = (lat[:,:y]-depth)%3
        elif y < 0:
            lat[:,y:] = (lat[:,y:]+depth)%3
        lat = np.roll(lat, -y, axis=1)
        if debug:
            print('lat post y translation',lat)

        hole_pos[0] = hole_pos[0] - step[1:]        #move 1st hole accordingly
        hole_pos[1] = hole_pos[1] - step[1:]        #move 2nd hole accordingly
        return lat, hole_pos
    
    def make_step(self, lat, hole_pos, step, D=0): 
        L_size = self.L_size
        if D != 0:
            L_size = 2*D+3

        n = step[0]                 
        hole_n_old = hole_pos[n]             #ith hole before step
        hole_n_new = hole_pos[n] + step[1:]  #ith hole after step  


        a = hole_pos[0] + step[1:]      
        b = np.array([1,1])
        if n == 0:
            b = hole_pos[1] - step[1:]  #why this? -> only for n = 0
        
        
        if any(a < 0) or any(a >= L_size) or any(b < 0) or any(b >= L_size):
            print(f'step={step} out of bounds')
            print(f'with lat:{lat}, hole_pos:{hole_pos}') 
            raise Exception('out of bounds')

        lat[hole_n_old[0], hole_n_old[1]],lat[hole_n_new[0], hole_n_new[1]] = lat[hole_n_new[0], hole_n_new[1]], lat[hole_n_old[0], hole_n_old[1]]  #switch x and y
        hole_pos[n] = hole_n_new     
        
        # mark both hole sites as zero
        for n in range(2):
            lat[hole_pos[n][0], hole_pos[n][1]] = 0     #marks both hole position as zero, relevant for first move where one hole switches with another site, overwriting the the other hole on central site
        
        if step[0] == 0 :
            lat, hole_pos = self.translation(lat, hole_pos, step, D=D)

        return lat, hole_pos

    def exchange_holes(self, state, debug=False):        #Linus 1.0

        lat = state['lat']
        hole_pos = state['hole_pos']
        seq = state ['seq']
        delta = hole_pos[1].copy()
        #switch hole positions
        hole_pos[1] = hole_pos[0].copy()
        hole_pos[0] = delta
        if debug:
            print(f'hole_pos:{hole_pos}')

        #translate back to center
        step = np.concatenate((np.zeros((1,), dtype=int), delta - self.depth - 1), axis=None)           #test translation() for step bigger than [1,1] done
        lat, hole_pos = self.translation(lat, hole_pos, step, debug=debug)
        state['lat'] = lat
        state['hole_pos'] = hole_pos
        if debug:
            print(f'step={step}')
            print(f'hole_pos after translation:{hole_pos}')
            print(f'lat after translation:{hole_pos}')

        #update seq
        seq2 = [] 
        for move in seq:
            move[0] = (move[0]+1)%2 
            seq2.append(move)
        state['seq'] = seq2 
        return state

    def generate_basis(self):
        if self.honeycomb:
            lattice = 'honeycomb'
            if self.big_unit_cell:
                uc = 'two site uc'
            else:
                uc = 'one site uc'
        else:
            lattice = 'triangular'
            if self.big_unit_cell:
                uc = 'three site uc'
            else:
                uc = 'one site uc'

        print(f'generate 2hole basis for {lattice} lattice with {uc}')
        if self.basis.length > 0:
            print('Basis has already been built')
        else:
            lat = self.Neel_state[0]
            hole_pos = [np.ones((2,), dtype=int) * (self.depth + 1) for _ in range(2)]
            seq = []
            state0 = {'lat': lat, 'hole_pos': hole_pos, 'seq': seq}
            for move in self.initial_moves:
                state_initial = copy.deepcopy(state0)
                self.generate_basis_element(state_initial, np.array(move, dtype=int))
            l = 1
            n0 = 0
            while l < self.depth:
                n1 = self.basis.length
                for n in range(n0, n1):
                    state0 = self.bin_basis[n]
                    for move in self.moves: 
                        if self.honeycomb == True:
                            # a = self.hole_is_on_2_sublattice({'seq': state0['seq']+[move]})
                            a = self.hole_is_on_2_sublattice(state0['seq']+[move])
                            if a == False:
                                state_initial = copy.deepcopy(state0)
                                self.generate_basis_element(state_initial, np.array(move, dtype=int))
                        else:
                            state_initial = copy.deepcopy(state0)
                            self.generate_basis_element(state_initial, np.array(move, dtype=int))
                l += 1
                n0 = n1

    def order_basis(self):
        ordered_list = []
        for x in self.basis.list:
            m = x[-1]
            ordered_list.append(self.bin_basis[m])
        self.bin_basis = ordered_list

    def generate_representatives(self):
        if self.basis.length == 0:
            raise Exception('basis has not been generated yet')
        self.is_representative = [(False, 0)] * self.basis.length   #list is longer then list of representatives (double as long?) -> as long as basis
        for n in range(self.basis.length):
            state = copy.deepcopy(self.bin_basis[n])
            state = self.exchange_holes(state)
            a = self.state_2_list_entry(state)
            found, m = self.basis.search(a)
            assert found
            # print(f'basis length: {self.basis.length}')
            # print(f'n:{n}, m:{m}')
            if m >= n:  # m=n impossible, holes cant be on top of each other
                self.is_representative[n] = (True, m, len(self.representatives))    
                self.representatives.append(self.bin_basis[n])
            else:
                self.is_representative[n] = (False, m, self.is_representative[m][2])

    def matrix_el(self):
    # compute matrix element of t-J-Hamiltonian up to a shift = energy of undoped Neel configuration
        ### use scipy.sparse matrix instead of np.array to reduce memory usage
        self.col_t = []
        self.row_t = []
        self.data_t = []
        self.col_t2 = []
        self.row_t2 = []
        self.data_t2 = []
        self.col_j = []
        self.row_j = []
        self.data_j = []
        self.col_V = []     #what is V? hole hole repulsion
        self.row_V = []
        self.data_V = []
        self.data_j_perp = []
        self.col_j_perp = []
        self.row_j_perp = []
        if self.honeycomb == False:
        # if self.honeycomb == self.honeycomb:
            # steps = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([1, 1, 1]),np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 1, 1])]
            steps = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([1, 1, 1]),np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 1, 1])]
            steps2 = [np.array([1, 1, 2]), np.array([1, 2, 1]), np.array([1, 1, -1]), np.array([0, 1, 2]), np.array([0, 2, 1]), np.array([0, 1, -1])] 
        else:
            steps = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([1, 1, 1]),np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 1, 1]), 
                     np.array([1, -1, 0]), np.array([1, 0, -1]), np.array([1, -1, -1]),np.array([0, -1, 0]), np.array([0, 0, -1]), np.array([0, -1, -1])] 
            steps2 = [np.array([1, 1, 2]), np.array([1, 2, 1]), np.array([1, 1, -1]), np.array([0, 1, 2]), np.array([0, 2, 1]), np.array([0, 1, -1]), 
                      np.array([1, -1, -2]), np.array([1, -2, -1]), np.array([1, -1, 1]), np.array([0, -1, -2]), np.array([0, -2, -1]), np.array([0, -1, 1])] 


        for i in range(len(self.representatives)):
        # for i in [4]:
        #     print(f'state {i}')
            state = self.representatives[i]
            lat = state['lat']
            hole_pos = state['hole_pos']
            seq = state['seq']
            sublattice = self.find_sublattice(state)
            lat0 = (lat-sublattice)%3
            x = hole_pos[0] 
            y = hole_pos[1] 
            lat0[x[0],x[1]] = 0
            lat0[y[0],y[1]] = 0
            diag = 0


            ####### compute H_{J_z}    
            ####### Groundstate Energy is chosen s.t. J_z = 0 if none of the holes moved 
            ####### maybe more physical to set it to Neel state Energy: +7*2/3*J
            ####### every satisfied bond in string has value 1/2*j; j value is multipiled in compute_H

            holes_x = hole_pos.copy()
            holes_y = hole_pos.copy()
            holes_diag = hole_pos.copy()
            lat_x = lat.copy()
            lat_y = lat.copy()
            lat_diag = lat.copy()
            lat_x, _ = self.translation(lat_x, holes_x, [0,1,0])
            lat_y, _ = self.translation(lat_y, holes_y, [0,0,1])
            lat_diag, _ = self.translation(lat_diag, holes_diag, [0,1,1])

            # counts all instances where neighbouring sites have the same value (includes hole sites, which is wrong)
            jx_sum = np.count_nonzero(lat - lat_x == 0)
            #print(f'state {i}: jx={jx_sum}')
            jy_sum = np.count_nonzero(lat - lat_y == 0)
            #print(f'state {i}: jy={jy_sum}')
            jdiag_sum = np.count_nonzero(lat - lat_diag == 0)
            #print(f'state {i}: jdiag={jdiag_sum}')
            diag = (jx_sum + jy_sum + jdiag_sum)/2

            # # remove contributions from links adjacent to one of the holes
            for n in range(2):
                xh = hole_pos[n]
                for move in [[1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,-1]]:
                    xh2 = xh + np.array(move)
                    if lat[xh2[0],xh2[1]] == lat[xh[0],xh[1]] and (xh2 != hole_pos[0]).any():   #second condlition avoids double counting if holes are adjacent
                        diag -= 1/2 
            
            self.data_j.append(diag)
            self.row_j.append(i)
            self.col_j.append(i)


            ### Calculate physical distance for V (Density-Density)
            phys_dist = np.zeros(2)
            hole_dist = hole_pos[1] - hole_pos[0]
            phys_dist[0], phys_dist[1] = np.sqrt(3)/2*hole_dist[0], hole_dist[1] - 1/2*hole_dist[0]
            abs_dist = np.sqrt(np.sum(phys_dist**2))

            if np.isclose((abs_dist), 1, atol=1e-6): #if the holes sit on neighboring sites
                self.data_V.append(1)
                self.row_V.append(i)
                self.col_V.append(i)


            ### compute H_{J_perp} part
            siteslist = (np.argwhere(lat0)).tolist()
            for site in siteslist: 
                nx = [[1,0],[0,1],[1,1]]  
                for nn in nx:
                    if [site[0]+nn[0],site[1]+nn[1]] in siteslist:
                        lat1 = lat.copy()  #is it faster to reverse lat1 by hand then to make new shallow copies? 
                        lat1[site[0],site[1]],lat1[site[0]+nn[0],site[1]+nn[1]]=lat1[site[0]+nn[0],site[1]+nn[1]],lat1[site[0],site[1]] #exchange both sites
                        state1 = {'lat': lat1, 'seq':seq, 'hole_pos': hole_pos}
                        a = self.state_2_list_entry(state1)
                        #print(a)
                        found, m  = self.basis.search(a)
                        rep, _, j = self.is_representative[m]
                        if found:
                            state_new = self.representatives[j]
                            lat_new=state_new['lat']
                            hole_pos_new=state_new['hole_pos']
                            seq_new=state_new['seq']
                            lat0_new = (lat_new - self.find_sublattice(state_new))%3
                            lat0_new[hole_pos[0][0],hole_pos[0][1]] = 0
                            lat0_new[hole_pos[1][0],hole_pos[1][1]] = 0
                            # print(f'TRI j_perp:{i} coupled to {j}')
                            # print(f'siteslist1:{siteslist}, hole_pos1:{hole_pos}, sl1:{self.find_hole_sublattice(seq)}')
                            # print(f'siteslist2:{(np.argwhere(lat0_new)).tolist()}, hole_pos2:{hole_pos_new}, sl2:{self.find_hole_sublattice(seq_new)}')
                            # print()
                            self.row_j_perp.append(j)
                            self.row_j_perp.append(i)
                            self.col_j_perp.append(i)
                            self.col_j_perp.append(j)
                            if rep:
                                self.data_j_perp.append((0,np.zeros((2,), dtype=int)))
                                self.data_j_perp.append((0,np.zeros((2,), dtype=int)))
                            else:
                                #print(f'i = {i}, j = {j}')
                                self.data_j_perp.append((1,-1*self.dist_2_phys_dist(hole_pos[1] - self.depth - 1, seq)))
                                self.data_j_perp.append((1,self.dist_2_phys_dist(hole_pos[1] - self.depth - 1, seq))) 
            
            ##### compute H_{t}(k) part:      Linus: test H_{k)}
            for step in steps:
                y1 = hole_pos[1]+step[1:]
                y2 = hole_pos[1]-step[1:]
                if np.all((y1 >= 0) & (y1 < self.L_size)) and np.all((y2 >= 0) & (y2 < self.L_size)):  
                    lat1 = lat.copy()
                    hole_pos1 = hole_pos.copy()
                    lat1, hole_pos1 = self.make_step(lat1, hole_pos1, step)
                    state1 = {'lat': lat1, 'hole_pos': hole_pos1, 'seq': seq + [step]}
                    # now state = H_{t}|i>
                    a = self.state_2_list_entry(state1)
                    found, m = self.basis.search(a)
                    if found:
                        (rep, _, j) = self.is_representative[m]
                        self.row_t.append(j)
                        self.col_t.append(i)
                        # print(f'TRI rep {i} couples to {j} via step: {step}')
                        if step[0] == 0: #first hole moves
                            if rep:
                                self.data_t.append((0, -1*self.dist_2_phys_dist(step[1:], seq))) #no holes exchanged, no displacement, but still hopping of first hole 
                                # print(f'{j} rep: {rep}, center hole moves')
                                # print(f'step: {self.dist_2_phys_dist(step[1:], seq)}, swapp distance: {self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1, seq)}')
                                # print(f'total phase={-self.dist_2_phys_dist(step[1:], seq)}')
                                # phase = -1*self.dist_2_phys_dist(step[1:], seq)
                                # print(f'Phase*K1={(phase[0]*K1[0] + phase[1]*K1[1])/np.pi}')
                                # print()                            
                            else:
                                self.data_t.append((1, -1*self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1 + step[1:], seq))) #holes exchanged + hopping 
                                # print(f'{j} rep: {rep}, center hole moves')
                                # print(f'step: {self.dist_2_phys_dist(step[1:], seq)}, swapp distance: {self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1, seq)}')
                                # print(f'total phase={-self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1 + step[1:], seq)}')
                                # phase = -1*self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1 + step[1:], seq)
                                # print(f'Phase*K1={(phase[0]*K1[0] + phase[1]*K1[1])/np.pi}')
                                # print()
                        else:   #second hole moves
                            if rep:
                                self.data_t.append((0, np.zeros((2,), dtype=int))) #no holes exchanged, no displacement
                                # print(f'{j} rep: {rep}, center hole doesnt move')
                                # print(f'step: {self.dist_2_phys_dist(step[1:], seq)}, swapp distance: {self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1, seq)}')
                                # print(f'total phase={0}')
                                # phase = 0
                                # print(f'Phase*K1={phase}')
                                # print()
                            else:
                                self.data_t.append((1, -1*self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1, seq))) #holes exchanged
                                # print(f'{j} rep: {rep}, center hole doenst move')
                                # print(f'step: {self.dist_2_phys_dist(step[1:], seq)}, swapp distance: {self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1, seq)}')
                                # print(f'total phase={-self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1, seq)}')
                                # # phase = -1*self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1, seq)
                                # # print(f'Phase*K1={(phase[0]*K1[0] + phase[1]*K1[1])/np.pi}')
                                # print()
        

            #### compute H_{t'}(k) part:   next nearest neighbour hopping
            for step in steps2:
                y1 = hole_pos[1]+step[1:]
                y2 = hole_pos[1]-step[1:]
                if np.all((y1 >= 0) & (y1 < self.L_size)) and np.all((y2 >= 0) & (y2 < self.L_size)):   
                    lat1 = lat.copy()
                    hole_pos1 = hole_pos.copy()                     
                    lat1, hole_pos1 = self.make_step(lat1, hole_pos1, step)
                    state1 = {'lat': lat1, 'hole_pos': hole_pos1, 'seq': seq + [step]}
                    y = hole_pos1[1]
                        # now state = H_{t}|i>
                    a = self.state_2_list_entry(state1)
                    found, m = self.basis.search(a)
                    if found:
                        # print('found t2 coupling')
                        # print(f'seq1: {seq}, step:{step}, seq2:{self.bin_basis[m]['seq']}')
                        # print()
                        (rep, _, j) = self.is_representative[m]
                        self.row_t2.append(j)
                        self.col_t2.append(i)
                        if step[0] == 0:
                            if rep:
                                self.data_t2.append((0, -1*self.dist_2_phys_dist(step[1:],seq))) #no holes exchanged, no displacement, but still hopping of first hole 
                                #print(f'i = {i}, seq_i={seq}, j = {j}, seq_j={self.representatives[j]['seq']}, step = {step}, rep')
                            else:
                                self.data_t2.append((1, -1*self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1 + step[1:],seq))) #distance in physical lattice
                        else:
                            if rep:
                                self.data_t2.append((0, np.zeros((2,), dtype=int))) #no holes exchanged, no displacement
                                #print(f'i = {i}, seq_i={seq}, j = {j}, seq_j={self.representatives[j]['seq']}, step = {step}, rep')
                            else:
                                self.data_t2.append((1, -1*self.dist_2_phys_dist(hole_pos1[1] - self.depth - 1,seq))) #distance in physical lattice  
         
    def compute_H(self, k, t=1, j=0.3, j_perp=0.3, t2=0, p=-1, V=0):
    # uses list of data points from matrix_el_j and momentum to create sparse matrix H
    # k (array of size (2,1)) = hole momentum in LLP-frame
        # print('len(data_t): ', len(self.data_t))    
        if len(self.data_t) > 0:
            data_1 = np.array([x[0] for x in self.data_t])
            data_2 = np.array([x[1] for x in self.data_t])
            data_t = t * p ** data_1 * np.exp(1j * (data_2[:,0]*k[0]+data_2[:,1]*k[1]))
        else:
            data_t = []
        

        if bool(t2) and len(self.data_t2) > 0:
            data_1 = np.array([x[0] for x in self.data_t2])
            data_2 = np.array([x[1] for x in self.data_t2])
            data_t2 = t2 * p ** data_1 * np.exp(1j * (data_2[:,0]*k[0]+data_2[:,1]*k[1]))            
            row_t2 = self.row_t2
            col_t2 = self.col_t2
        else:
            data_t2 = []
            row_t2 = []
            col_t2 = []
        # print('len data_t2 compute_H: ', len(data_t2))

        if bool(self.data_j_perp):
            data_1 = np.array([x[0] for x in self.data_j_perp])
            data_2 = np.array([x[1] for x in self.data_j_perp])
            data_j_perp = 1/2 * j_perp * p ** data_1 * np.exp(1j * (data_2[:,0]*k[0]+data_2[:,1]*k[1]))
            row_j_perp = self.row_j_perp
            col_j_perp = self.col_j_perp
        else:
            data_j_perp = []
            row_j_perp = []
            col_j_perp = []
        
        
        N = 1  # normalization factor for SU(3) t-J model
        if self.honeycomb == False:
            data = np.concatenate((data_t, np.conj(data_t), data_t2, np.conj(data_t2), N*j * np.array(self.data_j), N * np.array(data_j_perp), V * np.array(self.data_V)), axis=0)
            row = np.array(self.row_t + self.col_t + row_t2 + col_t2 + self.row_j + row_j_perp + self.row_V)
            col = np.array(self.col_t + self.row_t + col_t2 + row_t2 + self.col_j + col_j_perp + self.col_V)
        else:  
            data = np.concatenate((data_t, data_t2, j * np.array(self.data_j), np.array(data_j_perp), V * np.array(self.data_V)), axis=0)
            row = np.array(self.row_t + row_t2 + self.row_j + row_j_perp + self.row_V) 
            col = np.array(self.col_t + col_t2 + self.col_j + col_j_perp + self.col_V)

        self.data = data

        self.H = csr_matrix((data, (row,col)), shape=(len(self.representatives), len(self.representatives)), dtype=np.csingle)
        self.H.eliminate_zeros() # (only helpful if either t or j = 0)

        # --- Hermiticity Check ---
        diff = self.H - self.H.conj().T
        # We check if there are non-zero elements and if their magnitude exceeds a small tolerance 
        if diff.nnz > 0 and np.max(np.abs(diff.data)) > 1e-6:
            max_diff = np.max(np.abs(diff.data))
            raise ValueError(f"Hamiltonian self.H is not Hermitian! Maximum discrepancy: {max_diff}")
        # -------------------------

    def eigenval(self, state=0):
    # computes smallest eigenvalue of H
        Es, _ = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
        return np.sort(Es)[state]
    
    def eigenvec(self, state=0, v0=False):
    #computes eigenvector corresponding to the smallest eigenvalue
        if len(self.bin_basis) == 2:
            Es, vs = eigh(self.H.toarray())
            if Es[0] == Es[1]:
                return np.array([1, -1]) / np.sqrt(2)
            else:
                return vs[:, state]
        else:
            if v0:
                v0 = np.ones((len(self.bin_basis),))/np.sqrt(len(self.bin_basis))
                Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol, v0=v0)
            else:
                Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
            sort_ind = np.argsort(Es)
            vs = vs[:, sort_ind]
            return vs[:,state]
    
    def eigensys(self, state=0, full=False):
    # computes smallest eigenvalue of H
        # Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
        if self.depth > 2:
            Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
        else:
            Es, vs = eigh(self.H.toarray())
        sort_ind = np.argsort(Es)
        Es = Es[sort_ind]
        vs = vs[:, sort_ind]
        if full:
            return Es, vs
        else:
            
            return Es[state], vs[:,state]
        
    def eigensys(self, state=0, full=False, v0=None):
        if self.depth > 2:
            if isinstance(v0, str) and v0 == 'superposition':
                N_rep = len(self.representatives)
                v0 = np.ones(N_rep, dtype=np.complex64) / np.sqrt(N_rep)
            elif v0 is not None:
                v0 = np.asarray(v0, dtype=np.complex64)
            
            Es, vs = eigsh(self.H, k=state+1, which='SA', tol=self.tol, v0=v0)
        else:
            Es, vs = eigh(self.H.toarray())
        sort_ind = np.argsort(Es)
        Es = Es[sort_ind]
        vs = vs[:, sort_ind]
        if full:
            return Es, vs
        else:
            
            return Es[state], vs[:,state]
        
    def delete_weak_j_perp(self, t, j, j_perp, cutoff=1e-4):
        '''
        Looks for the j_perp processes with the most important contributions and delete the rest of them from self.data_j_perp
        '''
        assert len(self.row_j_perp) > 0
        important_processes = set()

        rot = np.array([[0, 1], [-1, 0]])

        k1 = np.array([0,0])
        k2 = np.array([1,1]) * np.pi
        k3 = np.array([1,0]) * np.pi
        k4 = np.array([1,1]) * np.pi / 2


        ks = [k2, k3, k4]
        for m in range(3 * 3):
            ks.append(rot @ ks[-3])
        ks = [k1] + ks

        vs = []
        for k in ks:
            self.compute_H(k, t, j, j_perp, p=-1)
            _, v = self.eigensys(0, full=True)
            v = v[:,0]
            vs.append(v)
        
        data_2 = np.array([x[1] for x in self.data_j_perp])

        exps = []
        for m, k in enumerate(ks):
            exp = np.array([np.conj(vs[m][self.row_j_perp[n]]) * vs[m][self.col_j_perp[n]] * np.exp(1j * np.sum(k * data_2[n,:])) for n in range(len(self.row_j_perp))])
            exps.append(exp)
        
        processes = np.amax(np.array([np.abs(exp1 - exp2) for exp1 in exps for exp2 in exps]), axis=0)

        # ###
        for x in np.arange(processes.size)[processes > cutoff]:
            important_processes.add(x)

        # add reflected state
        for x in list(important_processes):
            initial_state = self.representatives[self.col_j_perp[x]]
            final_state = self.representatives[self.row_j_perp[x]]
            # reflection along y axis
            mirrored_initial_state = self.mirror_y_state(initial_state)
            mirrored_final_state = self.mirror_y_state(final_state)
            found, m = self.basis.search(self.state_2_list_entry(mirrored_initial_state))
            assert found
            rep, _, j = self.is_representative[m]
            n_i = j
            found, m = self.basis.search(self.state_2_list_entry(mirrored_final_state))
            assert found
            rep, _, j = self.is_representative[m]
            n_f = j
            index = list(zip(self.col_j_perp, self.row_j_perp)).index((n_i, n_f))
            important_processes.add(index)

            # reflection along x axis
            mirrored_initial_state = self.mirror_x_state(initial_state)
            mirrored_final_state = self.mirror_x_state(final_state)
            found, m = self.basis.search(self.state_2_list_entry(mirrored_initial_state))
            assert found
            rep, _, j = self.is_representative[m]
            n_i = j
            found, m = self.basis.search(self.state_2_list_entry(mirrored_final_state))
            assert found
            rep, _, j = self.is_representative[m]
            n_f = j
            index = list(zip(self.col_j_perp, self.row_j_perp)).index((n_i, n_f))
            important_processes.add(index)

        # check if the corresponding rotated process is also included
        for x in list(important_processes):
            initial_state = self.representatives[self.col_j_perp[x]]
            final_state = self.representatives[self.row_j_perp[x]]

            index_2 = list(zip(self.col_j_perp, self.row_j_perp)).index((self.row_j_perp[x], self.col_j_perp[x]))
            important_processes.add(index_2)
            for _ in range(3):
                rotated_initial_state = self.rotate_state(initial_state)
                rotated_final_state = self.rotate_state(final_state)
                found, m = self.basis.search(self.state_2_list_entry(rotated_initial_state))
                assert found
                rep, _, j = self.is_representative[m]
                n_i = j
                found, m = self.basis.search(self.state_2_list_entry(rotated_final_state))
                assert found
                rep, _, j = self.is_representative[m]
                n_f = j

                rot_index = list(zip(self.col_j_perp, self.row_j_perp)).index((n_i, n_f))
                rot_index_2 = list(zip(self.col_j_perp, self.row_j_perp)).index((n_f, n_i))

                initial_state = copy.deepcopy(rotated_initial_state)
                final_state = copy.deepcopy(rotated_final_state)

                # print(x, rot_index)
                important_processes.add(rot_index)
                important_processes.add(rot_index_2)

        # add reverse state:
        for x in list(important_processes):
            index_2 = list(zip(self.col_j_perp, self.row_j_perp)).index((self.row_j_perp[x], self.col_j_perp[x]))
            important_processes.add(index_2)


        row = []
        col = []
        data = []
        for n in important_processes:
            row.append(self.row_j_perp[n])
            col.append(self.col_j_perp[n])
            data.append(self.data_j_perp[n])
        print(len(important_processes), len(processes))
        self.data_j_perp = data
        self.row_j_perp = row
        self.col_j_perp = col

    def dispersion(self,k_array,two_D=False, state=0, t=1, j=0.3, t2=0, j_perp=0.3, p=-1, V=0, v0=None):
        # returns array of energies corresponding to the moments in k_array
        # 2D == False: k_array = array of shape (Num_points,2)
        # 2D == True: k_array = Meshgrid(x,y)
    
        if two_D:
            print(f'Computing 2D dispersion for state {state}')
            Ev = []
            k_x=k_array[0]
            k_y=k_array[1]
            E=np.empty(k_x.shape)
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t=t, t2=t2, j=j, j_perp=j_perp, V=V, p=p) 
                    E[i,l],_=self.eigensys(state, v0=v0)

        else:
            # print(f'Computing 1D dispersion for state {state}')
            E=[] 
            Ev = []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t=t, t2=t2, j=j, j_perp=j_perp, p=p, V=V)
                es, vs = self.eigensys(state, v0=v0)
                E.append(es)
                Ev.append(vs)
        return np.array(E), np.array(Ev)

    def dispersion_nmax(self,k_array,two_D=False, num_n=1, t=1, t2=0, j=0.3, j_perp=0.3, p=-1, V=0):
    # returns array of energies corresponding to the moments in k_array
    # 2D == False: k_array = array of shape (Num_points,2)
    # 2D == True: k_array = Meshgrid(x,y)
        if two_D:
            k_x=k_array[0]
            k_y=k_array[1]
            Es=np.empty(k_x.shape + (num_n,))
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t=t, t2=t2, j=j, j_perp=j_perp, p=p, V=V)
                    E, _ = self.eigensys(num_n -1, full=True)
                    Es[i,l,:] = E
        else:
            E=[] 
            Evs = []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t=t, t2=t2, j=j, j_perp=j_perp, p=p, V=V)
                E.append(self.eigensys(num_n -1, full=True)[0])
                Evs.append(self.eigensys(num_n -1, full=True)[1])
        return np.array(E), np.array(Evs)
    
    def dispersion_nmax(self,k_array,two_D=False, num_n=1, t=1, t2=0, j=0.3, j_perp=0.3, p=-1, V=0):
        if two_D:
            k_x = k_array[0]
            k_y = k_array[1]
            # Initialize both containers matching the 2D grid dimensions
            Es = np.empty(k_x.shape + (num_n,))
            Evs = np.empty(k_x.shape + (num_n, len(self.representatives)), dtype=complex)
            
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l], k_y[i,l]], t=t, t2=t2, j=j, j_perp=j_perp, p=p, V=V)
                    E, V_sys = self.eigensys(num_n - 1, full=True)
                    Es[i,l,:] = E
                    Evs[i,l,:,:] = V_sys.T # Shape: (n_states, n_basis), stored to match grid
            return Es, Evs
        else:
            E=[] 
            Evs = []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t=t, t2=t2, j=j, j_perp=j_perp, p=p, V=V)
                E.append(self.eigensys(num_n -1, full=True)[0])
                Evs.append(self.eigensys(num_n -1, full=True)[1])
            return np.array(E), np.array(Evs)

    def rot_state_120(self, state):
        lat = state['lat']
        hole_pos = state['hole_pos']
        seq = state['seq']

        sublattice = self.find_sublattice(state)
        lat1 = (lat-sublattice)%3 
        lat1[hole_pos[0][0],hole_pos[0][1]] = 0 #set both hole positions to zero
        lat1[hole_pos[1][0],hole_pos[1][1]] = 0

        ## rotate lattice
        row, col = np.nonzero(lat1)         #find all flipped sites
        #print(row,col)
        val = lat[row,col]      
        rot_i = col - row + self.depth + 1      # New row index after rotation
        rot_j = -row + 2*(self.depth + 1)      # New column index after rotation
        #print(f'val{val},rot_i{rot_i},rot_j{rot_j}')
        rot_lat = sublattice.copy()
        rot_lat[rot_i, rot_j] = val

        ## rotate holes
        i = hole_pos[1][0]-self.depth-1
        j = hole_pos[1][1]-self.depth-1
        rot_hole = np.array([j-i+self.depth+1, -i+self.depth+1])
        

        ## rotate seq
        rot_seq = []
        for move in seq:
            move_rot = np.array([move[0], move[2]-move[1],-move[1]], dtype=int)
            rot_seq.append(move_rot)

        rot_state = {'lat': rot_lat, 'hole_pos': [hole_pos[0], rot_hole], 'seq': rot_seq}
        return rot_state

    def mirror_y_state(self, state):
        """Mirrors a state across the physical x-axis (sigma_y: rx -> -rx, ry -> ry)."""
        lat = state['lat']
        hole_pos = state['hole_pos']
        seq = state['seq']

        sublattice = self.find_sublattice(state)
        lat1 = (lat - sublattice) % 3
        lat1[hole_pos[0][0], hole_pos[0][1]] = 0
        lat1[hole_pos[1][0], hole_pos[1][1]] = 0

        ## 1. Mirror flipped spins on the lattice
        row, col = np.nonzero(lat1)
        val = lat[row, col]

        # Shift to center-relative coordinates
        i_rel = row - (self.depth + 1)
        j_rel = col - (self.depth + 1)

        # Apply (i, j) -> (-i, j - i)
        mir_i = -i_rel + (self.depth + 1)
        mir_j = (j_rel - i_rel) + (self.depth + 1)

        mir_lat = sublattice.copy()
        mir_lat[mir_i, mir_j] = val

        ## 2. Mirror moving hole position
        i_h = hole_pos[1][0] - (self.depth + 1)
        j_h = hole_pos[1][1] - (self.depth + 1)
        mir_hole = np.array([-i_h + (self.depth + 1), (j_h - i_h) + (self.depth + 1)], dtype=int)

        ## 3. Mirror move sequence
        mir_seq = []
        for move in seq:
            move_mir = np.array([move[0], -move[1], move[2] - move[1]], dtype=int)
            mir_seq.append(move_mir)

        return {'lat': mir_lat, 'hole_pos': [hole_pos[0], mir_hole], 'seq': mir_seq}

    def mirror_state(self, state, plane=0):
        """
        Computes the mirror reflection across plane n in {0, 1, 2}:
          plane=0: sigma_0 (y-axis reflection)
          plane=1: sigma_1 (rotated by 120 deg) = C3 @ sigma_0
          plane=2: sigma_2 (rotated by 240 deg) = C3^2 @ sigma_0
        """
        # First apply y-mirror
        mir_state = self.mirror_y_state(state)

        # Apply C3 rotations to access the other two mirror planes
        for _ in range(plane % 3):
            mir_state = self.rot_state_120(mir_state) #this convention leaves the uc = plane invariant under mirror symmetry

        return mir_state

    def build_mirror_matrix(self, k, plane=0, p=-1):
        """
        Builds the representation matrix M(k) for reflection symmetry in the representative basis.
        
        Parameters:
        k     : 2D momentum array [k_x, k_y]
        plane : 0 for sigma_y (y-axis reflection)
                1 for sigma_1 (rotated by 120 deg)
                2 for sigma_2 (rotated by 240 deg)
        p     : exchange parity (-1 for fermions, +1 for bosons)
        """
        row = []
        col = []
        data = []

        # 1. Momentum transformation matrix for plane 0 (sigma_y: x -> -x, y -> y)
        M_sigma_y = np.array([[-1.0,  0.0],
                            [ 0.0,  1.0]])

        # Clockwise C3 rotation matrix (-120 deg)
        theta = -2.0 * np.pi / 3.0
        C3 = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])

        # Full reflection matrix: sigma_plane = (C3)^plane @ sigma_y
        M_plane = np.linalg.matrix_power(C3, plane % 3) @ M_sigma_y
        k_mir = M_plane @ k

        # Base intra-unit-cell vector delta_0 for unit_cell=0 (from A at (0,0) to B at (0,1))
        delta_0 = np.array([0.0, 1.0])

        for n, state in enumerate(self.representatives):
            x, y = self.find_hole_sublattice(state['seq'])
            
            # Apply mirror reflection across the chosen plane
            mir_state = self.mirror_state(state, plane=plane)
            
            found, m = self.basis.search(self.state_2_list_entry(mir_state))
            if found:
                phase_mir = 0.0
                
                # Sublattice origin shift phase when central hole is on sublattice 1
                if self.honeycomb and self.big_unit_cell:
                    if x == 1:
                        delta = np.linalg.matrix_power(C3, self.unit_cell % 3) @ delta_0
                        dist = M_plane @ delta - delta
                        phase_mir = dist[0] * k_mir[0] + dist[1] * k_mir[1]

                (rep, _, m3) = self.is_representative[m]
                row.append(m3)
                col.append(n)

                if rep:
                    data.append(np.exp(-1j * phase_mir))
                else:
                    # Particle exchange phase with single parity sign 'p'
                    b = self.dist_2_phys_dist(mir_state['hole_pos'][1] - mir_state['hole_pos'][0], mir_state['seq'])
                    phase_exchange = b[0] * k_mir[0] + b[1] * k_mir[1]
                    data.append(p * np.exp(-1j * (phase_mir + phase_exchange)))
            else:
                print(f"Could not find mirror state for representative {n}")

        return csr_matrix((data, (row, col)), shape=(len(self.representatives), len(self.representatives)), dtype=complex)

    def rot_trial_state(self, m3, k, p=-1): #p is parity under particle exchange
        '''Computes unnormalized trial state with given rotational C3 eigenvalue m3 and momentum k'''
        assert len(self.representatives) > 0
        v = np.zeros((len(self.representatives)), dtype=complex)
        lat = self.Neel_state[0]
        hole_pos = [np.ones((2,), dtype=int) * (self.depth + 1) for _ in range(2)]
        seq = []
        state0 = {'lat': lat, 'hole_pos': hole_pos, 'seq': seq}
        steps = [[1, 1, 0], [1, 0, 1], [1, -1, -1]] #hole 1 moves from sl 0 to 1
        for n, step in enumerate(steps):
            step = np.array(step, dtype=int)
            state = copy.deepcopy(state0)
            lat, hole_pos = self.make_step(state['lat'], state['hole_pos'], step)
            seq = state['seq'] + [step]
            state = {'lat': lat, 'hole_pos': hole_pos, 'seq': seq}

            # search for this state in the basis of representatives
            found,m=self.basis.search(self.state_2_list_entry(state))
            rep, _, j = self.is_representative[m]
            if found:
                phase = 0
                phase_t = 0
                x,_ = self.find_hole_sublattice(state['seq'])
                a = np.sqrt(3)
                if self.big_unit_cell:
                    # phase = self.f(x,y)*a
                    phase_t = -self.g(x)*a
                if not rep:
                    a = self.dist_2_phys_dist(hole_pos[1] - self.depth - 1, seq)
                    phase = -1 * (a[0]*k[0]+a[1]*k[1]) + (p<0) * np.pi + phase_t
                v[j] += 1/np.sqrt(3) * np.exp(1j * m3 * n * 2*np.pi/3 + 1j*phase) 
        return v

    def f(self, x, y):
        return (-y)%3+(-x+1)%3+1
    
    def f_triangular(self, x, y):
        return {(0,0):0, (1,1):0, (2,2):0,
            (0,1):2, (0,2):1,
            (1,2):-1, (1,0):-2,
            (2,0):-1, (2,1):1}[(x, y)]

    def f_honeycomb(self,x,y): 
        return x-y
    
    def build_rot_matrix_old(self,k,p=-1): #change unit cell, st. green is in the middle
        row = []
        col = []
        data = []
        #k = 4*np.pi/(3*np.sqrt(3)) * np.array([1, 0]) * 0 #momentum k=0, why? because rotation should be momentum independent
        #self.compute_H(k=k)

        for n, state in enumerate(self.representatives):
        # for n in [1]: 
        #     state = self.representatives[n]
            #print(f'state {n}: {state}')
            x,y = self.find_hole_sublattice(state['seq'])
            phase_t = 0
            if not x == 0:
                state1 = state.copy()
                seq = state1['seq']
                lat = state1['lat']
                hole_pos = state1['hole_pos']
                f = (-x+1)%3-1      #map: f(0)=0, f(1)=-1, f(2)=1
                step = [0,f,0]
                #print(f'step={step}')
                lat1 = lat.copy()
                hole_pos1 = hole_pos.copy()
                lat, hole_pos = self.translation(lat1, hole_pos1, step)
                #print(f'step={step}') 
                state1['lat'] = lat
                state1['hole_pos'] = hole_pos
                state1['seq'] = seq + [step]
                #print(f'state after translation {n}: {state1}')
            else:
                state1 = state
            rot_state = self.rot_state_120(state1)        #perform 120degree rotation 
            #print(f'rot_state {n} pre translation: {rot_state}')
            if not x == 0:
                #print(f'x={x}')
                step_rot = np.array([step[0], step[1]-step[2],step[1]], dtype=int)
                #print(f'step_rot={step_rot}')
                rot_lat, rot_hole_pos = self.translation(lat, hole_pos, step_rot) 
                rot_state['lat'] = rot_lat
                rot_state['hole_pos'] = rot_hole_pos
                rot_seq = rot_state['seq']
                # print(f'seq pre pop:{rot_seq}')
                rot_seq.pop()
                # print(f'seq post pop:{rot_seq}')
                #print(f'rot_state {n} after translation: {rot_state}')
                dist = self.dist_2_phys_dist(step_rot[1:],seq)
                phase_t = k[0]*dist[0]+k[1]*dist[1]
            # print(f'state {n} after rot: {state}')
            # print()
            found, m = self.basis.search(self.state_2_list_entry(rot_state))
            if found:
                phase_r = 0
                a = k[0]*np.sqrt(3)
                if self.honeycomb and self.big_unit_cell:
                    # phase_r = self.f_honeycomb(x,y)*a
                    phase_r = -y*a   #why this phase? x=0, f(y)=-y
                elif not self.honeycomb and self.big_unit_cell:
                    # phase_r = self.f_triangular(x,y)*a 
                    phase_r = (-y)%3*a      #x=0, f(y)=mod_3(-y)
                (rep, _, m3) = self.is_representative[m]
                row.append(m3)
                col.append(n)
                if rep:
                    data.append(np.exp(1j*(phase_t+phase_r)))
                else:
                    a = self.dist_2_phys_dist(rot_state['hole_pos'][1] - rot_state['hole_pos'][0], rot_state['seq'])
                    data.append(np.real_if_close(p * np.exp(-1j*(a[0]*k[0]+a[1]*k[1] + phase_t+phase_r))))
            else:
                print(f'Couldrt find rot_state for state {n}')

        R = csr_matrix((data, (row, col)), shape=(len(self.representatives), len(self.representatives)))
        return R
    
    def build_rot_matrix(self, k, p=-1):
        row = []
        col = []
        data = []
        
        # 1. Match the clockwise rotation (-120 deg) of rot_state_120
        theta = -2 * np.pi / 3
        C3 = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
        k_rot = C3 @ k

        for n, state in enumerate(self.representatives):
            x, y = self.find_hole_sublattice(state['seq'])
            rot_state = self.rot_state_120(state) 
            found, m = self.basis.search(self.state_2_list_entry(rot_state))
            
            if found:
                phase_rot = 0
                if self.honeycomb:
                    # Sublattice shift phase when central hole is on sublattice 1
                    if x == 1:
                        dist = 1/2 * np.array([np.sqrt(3), -3])
                        if self.unit_cell == 0:
                            dist = dist
                        elif self.unit_cell == 1:
                            dist = C3 @ dist
                        elif self.unit_cell == 2:
                            dist = C3 @ C3 @ dist
                        phase_rot = dist[0] * k_rot[0] +dist[1] * k_rot[1]
                        
                (rep, _, m3) = self.is_representative[m]
                row.append(m3)
                col.append(n)
                
                if rep:
                    data.append(np.exp(-1j * phase_rot))
                else:
                    # Exchange phase with a single fermionic parity sign 'p'
                    b = self.dist_2_phys_dist(rot_state['hole_pos'][1] - rot_state['hole_pos'][0], rot_state['seq'])
                    phase_exchange = b[0] * k_rot[0] + b[1] * k_rot[1]
                    data.append(p * np.exp(-1j * (phase_rot + phase_exchange)))
            else:
                print(f'Could not find rot_state for state {n}')

        return csr_matrix((data, (row, col)), shape=(len(self.representatives), len(self.representatives)), dtype=complex)

    def test_c3_symmetry(self):
        for state in self.representatives:
            rotated_state = self.rot_state_120(state)
            a = self.state_2_list_entry(rotated_state)
            found, _ = self.basis.search(a)
            assert found, "Rotated state not found in basis!"
        return 'c3 symmetry tested'
        
# -----------------------------------------------------------------------------------

def run(args):
    depth = args["depth"]
    t = args["t"]
    t2 = args["t2"]
    j = args["j"]
    j_perp = args["j_perp"]
    ferm = args["fermions"]
    state = args["state"]
    connected = args["connected"]
    points_2D = args["points_2D"]
    grid_size = args["grid_size"]
    points_1D = args["points_1D"]
    honeycomb = args["honeycomb"]
    unit_cell = args["unit_cell"]
    D1 = args["1D_disp"]
    D2 = args["2D_disp"]
    all_2D_bands = args["all_2D_bands"]
    Magnetic_BZ = args["Magnetic_BZ"]


    ### Create string basis
    t0 = perf_counter()
    sb = StringBasis(depth, connected, honeycomb, unit_cell)
    print('coefficients:',depth, j, j_perp, t, t2, connected, state, grid_size, points_2D, points_1D, honeycomb, unit_cell)
    print('created String Basis in {t:.3f}s \n'.format(t=perf_counter()-t0))
    
    t0 = perf_counter()
    sb.matrix_el()
    print('computed matrix element in {t:.3f}s'.format(t=perf_counter()-t0))
    
    # make momentum arrays
    # 2D
    k_vals = np.linspace(-grid_size, grid_size, points_2D)
    k_grid = np.meshgrid(k_vals, k_vals)

    # 1D
    K = 4*np.pi/(3*np.sqrt(3))*np.array([1, 0])
    Kp = 2*np.pi/(3*np.sqrt(3))*np.array([1, np.sqrt(3)])
    M = np.pi/3*np.array([np.sqrt(3), 1])
    Gamma = np.array([0,0])

    # Paths between symmetry point 
    path1 = np.linspace(Gamma, K, int(points_1D/3), endpoint=False)
    path2 = np.linspace(K, M, int(points_1D/6), endpoint=False)
    path3 = np.linspace(M, Kp, int(points_1D/6), endpoint=False)
    path4 = np.linspace(Kp, Gamma, int(points_1D/3)+1)
    k_path = np.vstack((path1, path2, path3, path4))

    path_data = '/Users/linushein/Documents/Python/Python_output/SU(3)_truncated2/'

    if ferm == True:
        stat = 'fermion'
        p = -1
    else:
        stat = 'boson'
        p = 1

    if D1 == True:
        t0 = perf_counter()
        E_1D = np.zeros([state,len(k_path)])
        for i in range(state):
            E_1D[i], _ = sb.dispersion(k_path, two_D=False, state = i, t=t, t2=t2, j=j, j_perp=j_perp, p=p)
            #print(f'band {i}: E = {E_1D[i]}')
        print('computed 1D dispersion in {t:.3f}s'.format(t=perf_counter()-t0))
        if honeycomb == True:
            name2 = f'string_2hole_1D_disp_SU2_Honeycomb_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands_{stat}.npy'
        else:
            name2 = f'string_2hole_1D_disp_SU(3)_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands_{stat}.npy'
        np.save(os.path.join(path_data, name2), E_1D)
        #print(E_1D)

    if D2 == True:
        t0 = perf_counter()
        x = copy.deepcopy(state)
        if all_2D_bands == False:
            x = 1
        E_2D = np.zeros([x,points_2D,points_2D])
        # print(f'E2D shape = {E_2D.shape}')
        for i in range(x):
            E_2D[i], _ = sb.dispersion(k_grid, t=t, t2=t2, j=j, j_perp=j_perp, p=p, two_D=True, state=i)
        print('computed 2D dispersion in {t:.3f}s'.format(t=perf_counter()-t0))
        #print(f'E2D shape = {E_2D.shape}')
        if honeycomb == True:
            name = f'string_2hole_2D_disp_SU2_Honeycomb_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands_{stat}.npy'
        else:
            name = f'string_2hole_2D_disp_SU(3)_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands_{stat}.npy'            
        np.save(os.path.join(path_data, name), E_2D)

if __name__ == "__main__":
    args = {
        "depth": 8,
        "j": 0.30,
        "j_perp": 0.30,
        "t": 1.0,
        "t2": 0,
        "fermions": True,
        "connected": True,  
        "state": 6,
        "grid_size": 3.3,  
        "points_2D": 67,
        "points_1D": 180,
        "honeycomb": True,
        "unit_cell": True,
        "1D_disp": True,
        "2D_disp": True,
        "all_2D_bands": False,
        "Magnetic_BZ": True
    }
    run(args)

