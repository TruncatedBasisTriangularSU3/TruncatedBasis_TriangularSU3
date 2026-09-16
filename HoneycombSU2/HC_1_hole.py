import numpy as np
import copy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

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

class StringBasisHC:
# A class for generating a truncated basis
    def __init__(self, depth, only_connected=True, initial_sl=0):  # works only for depth<=14 otherwise change uint32 to uint64!
        self.depth = depth
        self.L_size = 2*self.depth+3
        self.basis = sorted_list(length_arr = self.L_size)
        self.bin_basis=[]
        #self.bin_basis_ordered = []
        self.last_move = [None, None]
        self.moves = [[[-1, 0], [0, -1], [0, 1]],[[1,0],[0,-1],[0,1]]] # 0: even sites, particle can move up, 1: odd sites, particle can move down
        self.H = None
        self.rot_mat = None
        self.initial_sl = initial_sl

        self.only_connected = only_connected
        
        self.tol = 0

        self.generate_basis()
        self.order_basis()
        self.matrix_el()
        #self.calc_rot_mat()

    def connected(self, state): #THIS DOES NOT WORK YET
        # state should be list of dictionary with keys 'lat', 'sl', where sl denotes A or B sublattice
        res = False
        a = state['lat']
        sl = state['sl']
        # construct list of sites with flipped spins
        flipped = (np.argwhere(a)).tolist()
        # determine if the flipped spins are connected
        if len(flipped) == 0:
            res = True
        else:
            flipped = flipped + [[self.depth+1]*2]
            moves = list(self.moves)
            comp = [flipped.pop()]
            n = 0
            while len(flipped) > 0 and n < len(comp):
                r = comp[n]
                n += 1
                pos_moves = list(moves[sl])
                for move in moves:
                    testsite = list(np.array(r) + move)
                    if testsite in flipped:
                        comp.append(flipped.pop(flipped.index(testsite)))
                sl = (sl+1)%2
            res = len(flipped) == 0
            # if not res:
            #     print(state, 'is not connected')
        return res

    def state_2_list_entry(self, state):
        """ converts 2d array of True/False entries to list of numbers, correspondig to binary representation of each row """
        lat = state['lat']
        a = lat*np.matmul(np.ones((self.L_size, 1), dtype=np.uint32),
                          np.reshape(2**np.arange(self.L_size, dtype=np.uint32), (1, self.L_size)), dtype=np.uint32)
        list_entry = np.sum(a, axis=1).tolist()
        return list_entry

    def generate_basis_element(self, state, step):
    # an old state and a step this function generetates a basis element and adds it to the basis list
    # state = dict()
    # seq is a list of steps = np.arrays where seq[i] = [x, y]
    # x,y in {-1, 0, 1} denote the direction of the hopping
        lat = self.make_step(state['lat'], step)
        sl = (state['sl'] + 1)%2 #each hop changes the sublattice
        state = {'lat': lat, 'sl': sl}
        #print("state=", state)
        if self.only_connected:
            physical = self.connected(state)
        else:
            physical = True
        new = False
        if physical:
            new = self.basis.add(self.state_2_list_entry(state) + [self.basis.length])
        check = new and physical
        #print(state)
        if check:
            self.bin_basis.append(state)
    
    def translation(self, lat, step):
        for n in range(2):
            lat = np.roll(lat, -step[n], axis=n)
        return lat
    
    def make_step(self, lat, step):
        # lat = 2D boolean array with spin configurations (holes are False)
        # step = [x, y] gives the hopping
        x = np.ones((2,), dtype=int)* (self.depth +1) # position of the hole
        y = x + step #new hole position
        if np.sum(np.abs(step)) % 2 == 1: #hole jumps and uneven number of steps in total => spin at center opposite from new hole site?
            lat[x[0], x[1]] = not(lat[y[0], y[1]])
        else: #hole jumps even number of steps =>  new value at center same as value at new hole position?
            lat[x[0], x[1]] = lat[y[0], y[1]]

        # mark hole site as False
        lat[y[0], y[1]] = False

        lat = self.translation(lat, step)
        #lat = np.flip(lat, axis=0) #flip along y to ensure central site

        return lat

    # def make_step_unitcell(self, lat, step):
    #     # lat = 2D boolean array with spin configurations (holes are False)
    #     # step = [x, y] gives the hopping
    #     x = np.ones((2,), dtype=int)* (self.depth +1) # position of the hole
    #     y = x + step #new hole position
    #     if np.sum(np.abs(step)) % 2 == 1: #hole jumps and uneven number of steps in total => spin at center opposite from new hole site?
    #         lat[x[0], x[1]] = not(lat[y[0], y[1]])
    #     else: #hole jumps even number of steps =>  new value at center same as value at new hole position?
    #         lat[x[0], x[1]] = lat[y[0], y[1]]

    #     # mark hole site as False
    #     lat[y[0], y[1]] = False

    #     return lat

    def generate_basis(self):
        """ method to generate the entire basis (as list of 1D list of uint32) """
        lat = np.zeros((self.L_size, self.L_size), dtype=bool)
        #Neel state, 0=A sublattice: connection to upper site
        #=> hole on even site, i.e. all even sites have connection to upper site, odd to lower site
        state0 = {'lat': lat, 'sl': self.initial_sl}
        #print("state0", state0)

        self.basis.add(self.state_2_list_entry(state0) + [self.basis.length]) #adds state and position in unordered bin basis as last element
        self.bin_basis.append(state0)

        l = 0
        n0 = 0
        while l < self.depth:
            n1 = self.basis.length
            for n in range(n0, n1):
                state0 = self.bin_basis[n]
                sl = state0['sl']
                #print(sl)
                for move in self.moves[sl]:
                    #print("move=", move)
                    state_initial = copy.deepcopy(state0)
                    self.generate_basis_element(state_initial, np.array(move, dtype=int))
            l += 1
            n0 = n1

    def order_basis(self):
        #orders the bin basis
        ordered_list = []
        for x in self.basis.list:
            m = x[-1] #after this is used for basis ordering, this last entry has no meaning anymore
            ordered_list.append(self.bin_basis[m])
        self.bin_basis = ordered_list

    def matrix_el(self):
    # compute matrix element of t-J-Hamiltonian up to a shift = energy of undoped Neel configuration
        ### use scipy.sparse matrix instead of np.array to reduce memory usage
        self.col_t1 = []
        self.row_t1 = []
        self.data_t1 = []
        self.col_t2 = []
        self.row_t2 = []
        self.data_t2 = []
        self.col_t3 = []
        self.row_t3 = []
        self.data_t3 = []
        self.col_j0 = []
        self.row_j0 = []
        self.data_j0 = []
        self.col_j2 = []
        self.row_j2 = []
        self.data_j2 = []


        for i, state in enumerate(self.bin_basis):
            lat = state['lat']
            sl = state['sl']

            ####### compute diagonal part of H_J
            #bond to the top: even sites if hole on sublattice A (sl=0), odd if B (sl=1)
            latsum_up = np.logical_xor(lat, np.roll(lat, -1, axis=0)).flatten()
            latsum_up[(sl)::2]=0 #set indices with no upper connection to zero
            diag = np.sum(latsum_up)
            
            #horizontal conectivity as for square
            diag += np.sum(np.logical_xor(lat, np.roll(lat, 1, axis=1)))
            
            # remove contributions from links adjacent to one of the holes
            xh = yh = self.depth + 1
            for x in [-1, 1]:
                if lat[yh, x+xh]:
                    diag -= 1
            if sl==0:
                y = -1
                if lat[yh+y, xh]:
                    diag -= 1
            elif sl==1:
                y = 1
                if lat[yh+y, xh]:
                    diag -= 1

            self.data_j0.append(diag)
            self.row_j0.append(i)
            self.col_j0.append(i)

            ##### compute off-diagonal part of H_J
            siteslist = list(np.argwhere(lat))

            for site in siteslist:
                lat1=lat.copy()
                ### bonds along x direction
                if lat1[site[0],site[1]] and lat1[site[0],site[1]+1]:
                    lat1[site[0],site[1]+1] = not lat1[site[0],site[1]+1]
                    lat1[site[0],site[1]] = not lat1[site[0],site[1]]
                    a = self.state_2_list_entry({'lat': lat1.copy()})
                    found, j = self.basis.search(a)
                    if found:
                      self.row_j2.append(j)
                      self.col_j2.append(i)
                      self.data_j2.append(1)
                    lat1[site[0],site[1]] = not lat1[site[0],site[1]]
                    lat1[site[0],site[1]+1] = not lat1[site[0],site[1]+1]
            
                ### bonds along y direction
                if (self.L_size*site[0]+site[1]+sl)%2==0: #bond upwards
                    if lat1[site[0],site[1]] and lat1[site[0]-1,site[1]]:
                        lat1[site[0],site[1]] = False
                        lat1[site[0]-1,site[1]] = False
                        a = self.state_2_list_entry({'lat': lat1.copy()})
                        found, j = self.basis.search(a)
                        if found:
                            self.row_j2.append(j)
                            self.col_j2.append(i)
                            self.data_j2.append(1)

            ##### compute H_{t}(k) part:
            ### along x direction as for square lattice, but only take hopping from sublattice A
            if sl==0:
                step1 = np.array([0, 1]) #hop from upper site to the left (along a1)
                lat1 = lat.copy()
                lat1 = self.make_step(lat1, step1)
                state1 = {'lat': lat1, 'sl': 1}
                # now state = H_{t}|i>
                a = self.state_2_list_entry(state1)
                found, j = self.basis.search(a)
                if found:
                    self.row_t1.append(j)
                    self.col_t1.append(i)
                    self.data_t1.append(1)

                step2 = np.array([0, -1]) #hop from upper site to the left (along a2)
                lat2= lat.copy()
                lat2 = self.make_step(lat2, step2)
                state2 = {'lat': lat2, 'sl': 1}
                # now state = H_{t}|i>
                a = self.state_2_list_entry(state2)
                found, j = self.basis.search(a)
                if found:
                    self.row_t2.append(j)
                    self.col_t2.append(i)
                    self.data_t2.append(1)
                  
                ### along y direction: upwards if hole on sl 0 (down if on sl 1)
                step3 = np.array([-1, 0]) #hops within unit cell
                lat3 = lat.copy()
                lat3 = self.make_step(lat3, step3)
                state3 = {'lat': lat3, 'sl': 1}
                # now state = H_{t}|i>
                a = self.state_2_list_entry(state3)
                found, j = self.basis.search(a)
                if found:
                    self.row_t3.append(j)
                    self.col_t3.append(i)
                    self.data_t3.append(1)

        self.data_t1 = np.array(self.data_t1) #hop left
        self.data_t2 = np.array(self.data_t2) #hop right
        self.data_t3 = np.array(self.data_t3) #hop up
        #print(self.data_t1, self.data_t2, self.data_t3)
        self.data_j0 = np.array(self.data_j0)
        self.data_j2 = np.array(self.data_j2)


    def compute_H(self, k, t, j):
    # uses list of data points from matrix_el_j and momentum to create sparse matrix H
    # k (array of size (2,1)) = hole momentum in LLP-frame, in x,y basis, convert to k1 and k2
        #b1 = 2*np.pi/np.sqrt(3)*(1,-1/np.sqrt(3))
        #b2 = 2*np.pi/np.sqrt(3)*(-1,-1/np.sqrt(3))
        kx = k[0]
        ky = k[1]
        #upper right matrix (h_kA^dag,h_kB)
        if len(self.data_t1) > 0:
            data_t = np.concatenate((t * np.exp(complex(0,-1/2*(np.sqrt(3)*kx-3*ky))) * self.data_t1,
                                     t * np.exp(complex(0,-1/2*(-np.sqrt(3)*kx-3*ky))) * self.data_t2,
                                     t * np.exp(complex(0,0)) * self.data_t3), axis=0)
            
            # data_t = np.concatenate((t * np.exp(complex(0,-1/2*(np.sqrt(3)*kx-ky))) * self.data_t1,
            #                          t * np.exp(complex(0,-1/2*(-np.sqrt(3)*kx-ky))) * self.data_t2,
            #                          t * np.exp(complex(0,-ky)) * self.data_t3), axis=0)
      
      
            row_t = self.row_t1 + self.row_t2 + self.row_t3
            col_t = self.col_t1 + self.col_t2 + self.col_t3
            #print(self.data_t1, self.row_t1, self.col_t1,np.exp(complex(0,-1/2*(np.sqrt(3)*kx-ky))), np.angle(np.exp(complex(0,-1/2*(np.sqrt(3)*kx-ky))))/np.pi)
        
        else:
            data_t = []
            row_t = []
            col_t = []


        #j on diagonals ? for now also off diagonal but not sure if this is true
        data_j_diag =  j/2 * self.data_j0
        row_j_diag = self.row_j0 
        col_j_diag = self.col_j0
        
        data_j_offdiag = j / 2 * self.data_j2
        row_j_offdiag = self.row_j2
        col_j_offdiag = self.col_j2


        #h1 = np.concatenate((data_t,
                               #data_j_diag,
                               #data_j_offdiag, np.conj(data_j_offdiag)), axis=0)
        #h2 = np.concatenate((data_t.conj(),
                               #data_j_diag,
                               #data_j_offdiag, np.conj(data_j_offdiag)), axis=0)

        data = np.concatenate((data_t, np.conj(data_t),
                               data_j_diag,
                               data_j_offdiag, np.conj(data_j_offdiag)), axis=0)
        
        #print(data)
        row = np.array(row_t + col_t + row_j_diag + row_j_offdiag + col_j_offdiag)
        #print(row)
        col = np.array(col_t + row_t + col_j_diag + col_j_offdiag + row_j_offdiag)
        #print(col)
        self.H = csr_matrix((data, (row,col)), shape=(self.basis.length, self.basis.length), dtype=np.csingle)
        #H1 = csr_matrix((h1, (row,col+self.basis.length)), shape=(self.basis.length, self.basis.length), dtype=np.csingle)
        #H2 = csr_matrix((h1, (row+self.basis.length,col)), shape=(self.basis.length, self.basis.length), dtype=np.csingle)
        #self.H = = scipy.sparse.bmat([[None, H1], [H2, None]], format='csr')
        
        self.H.eliminate_zeros() # (only helpful if either t or j = 0)

    def rot_trial_state(self, m3, k):
        '''Computes unnormalized trial state with given rotational C6 eigenvalue m6 and momentum k'''
        lat = np.zeros((self.L_size, self.L_size), dtype=bool)
        #Neel state, 0=A sublattice: connection to upper site
        #=> hole on even site, i.e. all even sites have connection to upper site, odd to lower site
        state0 = {'lat': lat, 'sl': 0}
        v = np.zeros((self.basis.length), dtype=complex)
        lat = np.zeros((self.L_size, self.L_size), dtype=bool)
        steps = [[0, 1], [0, -1], [-1, 0]] #hole 0 moves from sl 0 to 1
        for n, step in enumerate(steps):
            step = np.array(step, dtype=int)
            state = copy.deepcopy(state0)
            lat = self.make_step(state['lat'], step)
            sl = (state['sl'] + 1)%2 #each hop changes the sublattice
            state = {'lat': lat, 'sl': sl}

            # search for this state in the basis
            a = self.state_2_list_entry(state)
            found, j = self.basis.search(a)
            if found:
                v[j] += 1/(np.sqrt(3)) * np.exp(1j * m3 * n * 2*np.pi/3-1j*1/2*(np.sqrt(3)*k[0]+3*k[1])*n)
        return v

    def eigenval(self, state=0):
    # computes smallest eigenvalue of H
        # ARPACK requires k < N - 1. Here k = state + 1, and N = self.basis.length.
        if self.basis.length > state + 2:
            Es, _ = eigsh(self.H, k=state+1, which='SA', tol=self.tol)
        else:
            Es, _ = eigh(self.H.toarray())
        return np.sort(Es)[state]
    
    def eigenvec(self, state=0, v0=False):
    #computes eigenvector corresponding to the smallest eigenvalue
        # if len(self.representatives) == 2:
        #     Es, vs = eigh(self.H.toarray())
        #     if Es[0] == Es[1]:
        #         return np.array([1, -1]) / np.sqrt(2)
        #     else:
        #         return vs[:, state]
        # else:
        if v0:
            v0 = np.ones((len(self.representatives),))/np.sqrt(len(self.representatives))
            Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol, v0=v0)
        else:
            Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
        sort_ind = np.argsort(Es)
        vs = vs[:, sort_ind]
        return vs[:,state]
    
    def eigensys(self, state=0, full=False):
    # computes smallest eigenvalue of H
        try:
            Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
        except:
            Es, vs = np.linalg.eigh(self.H.toarray())
            Es = Es[:state+1]
            vs = vs[:,:state+1]
        sort_ind = np.argsort(Es)
        Es = Es[sort_ind]
        vs = vs[:, sort_ind]
        if full:
            return Es, vs
        else:
            
            return Es[state], vs[:,state]

    def dispersion(self, k_array,two_D=False, state=0, t=1, j=0.3):
    # returns array of energies corresponding to the moments in k_array
    # 2D == False: k_array = array of shape (Num_points,2)
    # 2D == True: k_array = Meshgrid(x,y) ks in x,y basis
        if two_D:
            k_x=k_array[0]
            k_y=k_array[1]
            E=np.empty(k_x.shape)
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t=t, j=j) 
                    E[i,l]=self.eigenval(state)

        else:
            E=[] 
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t=t, j=j)
                E.append(self.eigenval(state))
        return np.array(E)

    def dispersion_nmax(self,k_array,two_D=False, num_n=1, t=1, j=0.3):
    # returns array of energies corresponding to the moments in k_array
    # 2D == False: k_array = array of shape (Num_points,2)
    # 2D == True: k_array = Meshgrid(x,y)
        if two_D:
            k_x=k_array[0]
            k_y=k_array[1]
            Es=np.empty(k_x.shape + (num_n,))
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t, j)
                    E, _ = self.eigensys(num_n -1, full=True)
                    Es[i,l,:] = E
        else:
            E=[]
            evs = []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t, j)
                E.append(self.eigensys(num_n -1, full=True)[0])
                evs.append(self.eigensys(num_n -1, full=True)[1])
                
        return np.array(E), np.array(evs)


    def brick_to_hc(self, x,y, sl):
        d = x-y
        n1 = -y
        n2 = y + int(d/2) + np.sign(d)*np.abs((2*(sl-0.5)-np.sign(d))//2*((d)%2))
        n3 = -int(d/2) - np.sign(d)*np.abs((2*(sl-0.5)+np.sign(d))//2*((d)%2))
        return int(n1), int(n2), int(n3)

    def hc_to_brick(self, n1,n2,n3,sl):
        y = -n1
        d = n2 - n3 - y
        x = y + d
    
        n1_check, n2_check, n3_check = self.brick_to_hc(x, y, sl)
        if (n2_check == n2) and (n3_check == n3):
            return int(x), int(y)
    
        raise ValueError("No valid (x, y, sl) found for the given (n1, n2, n3)")

    
    def rotate_state(self, state):
        lat = state['lat']
        sl = state['sl']
        depth = self.depth
        new_lat = np.zeros(lat.shape, dtype=bool)
        
        for site in np.argwhere(lat):
            x = site[1]-depth-1
            y = site[0]-depth-1
            #print(x,y)
            n1, n2, n3 = self.brick_to_hc(x,y,sl)
            #print(n1,n2,n3)
            new_x, new_y =  self.hc_to_brick(n3,n1,n2,sl)
            #print(new_x, new_y)
            new_lat[new_y+depth+1,(new_x+depth+1)] = True
        new_state = {'lat': new_lat, 'sl': sl}
        return new_state

    def calc_rot_mat(self, k):
        Rm = np.zeros((self.basis.length, self.basis.length), dtype=complex)

        for j, state in enumerate(self.bin_basis):
            #fig, axs = plt.subplots(1,2, figsize=(4,2))
            #axs[0].imshow(state['lat'])
            state_rot = self.rotate_state(state)
            #axs[1].imshow(state_rot['lat'])
            #plt.show()
            a = self.state_2_list_entry(state_rot)
            found, i = self.basis.search(a)
            l = np.sum(state_rot['lat'])
            if l%2 != 0:
                Rm[i,j] = 1*np.exp(-1j*1/2*(np.sqrt(3)*k[0]+3*k[1]))
            else:
                Rm[i,j] = 1
        self.rot_mat = Rm
            