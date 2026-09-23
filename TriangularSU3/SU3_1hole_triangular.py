import numpy as np
import copy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from time import perf_counter
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 20,
    "legend.fontsize": 15,
    "savefig.dpi": 300,
    # "axes.grid": True,
    "axes.axisbelow": True
})

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
        self.Neel_state = []
        self.basis = sorted_list(length_arr = self.L_size)
        self.bin_basis=[]
        self.last_move = [None, None]
        self.moves = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]
        self.H = None

        self.honeycomb = honeycomb
        self.only_connected = only_connected

        self.unit_cell = unit_cell
        if unit_cell is None:
            self.big_unit_cell = False
        elif isinstance(unit_cell, int):
            self.big_unit_cell = True
        print(f'1hole big_unit_cell={self.big_unit_cell}, unit_cell={unit_cell}')

        self.tol = 0

        self.triangular_Neel()
        self.generate_basis()
        self.order_basis()
        self.matrix_el()


        self.data_t_test = []
        self.data = []

    def triangular_Neel(self):
        # generates the three possible neel states (center site = 0,1,2 i.e red, green, blue) and saves them to list, because they are needed often
        for k in range(3):
            triangular_lattice = np.zeros((self.L_size, self.L_size), dtype=int)
            for i in range(self.L_size):
                for j in range(self.L_size):
                    if (i+j+k+self.depth+1) % 3 == 0: # +self.depth+1 so center site is same for all lattice sizes
                        triangular_lattice[i,j] = 0
                    elif (i+j+k+self.depth+1) % 3 == 1:
                        triangular_lattice[i,j] = 1
                    else:
                        triangular_lattice[i,j] = 2
            triangular_lattice[self.depth+1,self.depth+1] = 0
            #print(triangular_lattice)
            self.Neel_state.append(triangular_lattice)
    
    def find_sublattice(self,state):
        seq = state['seq']
        y = np.sum(np.array(seq), axis=0, dtype=int)
        y = (np.sum(y))%3
        sublattice = self.Neel_state[y]
        return sublattice
    
    def find_hole_sublattice(self,seq): 
        x = np.sum(np.array(seq), axis=0, dtype=int)
        x = (np.sum(x))%3
        return x
    
    def hole_is_on_2_sublattice(self,state):
        innit = False
        seq = state['seq']
        x = np.sum(np.array(seq), axis=0, dtype=int)
        x = (np.sum(x))%3
        if x == 2:
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
            z = np.sum(np.array(seq), axis=0, dtype=int)
            z = (-np.sum(z)+1)%3
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
            z = np.sum(np.array(seq), axis=0, dtype=int)
            z = (-np.sum(z)+1)%3
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
            z = np.sum(np.array(seq), axis=0, dtype=int)
            z = (-np.sum(z)+1)%3
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

    def connected(self, state): 
        # state should be list of dictionary with keys 'lat', 'seq'
        res = False
        lat = state['lat']      
        sublattice = self.find_sublattice(state)
        lat = (lat-sublattice)%3 #subtract background
        lat[self.depth+1,self.depth+1] = 0 #mark hole site as zero again, redundant? check
        # construct list of sites with flipped spins
        flipped = (np.argwhere(lat)).tolist()
        # determine if the flipped spins are connected
        if len(flipped) == 0:
            res = True
        else:
            flipped = flipped + [[self.depth+1]*2] #add hole position to list
            moves = list(np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]))
            comp = [flipped.pop()] #removes hole position from list again, cumbersome way to save hole position as comp
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

    def state_2_list_entry(self, state):     #changed to ternary rep
        """ converts 2d array of True/False entries to list of numbers, correspondig to binary representation of each row """ 
        lat = state['lat']      
        sublattice = self.find_sublattice(state)
        #print(f'lat: {lat}')
        #print(f'neel: {sublattice}')
        lat = (lat-sublattice)%3

        #mark hole site as zero
        y = np.ones((2,), dtype=int)* (self.depth +1)
        lat[y[0], y[1]] = 0
        # print(seq)
        # print('lat minus background:')
        # print(lat)

        a = lat*np.matmul(np.ones((self.L_size, 1), dtype=np.uint64),
                          np.reshape(3**np.arange(self.L_size, dtype=np.uint64), (1, self.L_size)), dtype=np.uint32)
        list_entry = np.sum(a, axis=1).tolist()
        return list_entry 
    
    def translation(self, lat, step): #done
        for n in range(2):
            if n == 0:
                if step[n] == 1:
                    lat[0, :] = (lat[0, :]-self.depth)%3
                elif step[n] == -1:
                        lat[-1, :] = (lat[-1, :]+self.depth)%3
                lat = np.roll(lat, -step[n], axis=n)
            elif n == 1:
                if step[n] == 1:
                    lat[:, 0] = (lat[:,0]-self.depth)%3
                elif step[n] == -1:
                    lat[:, -1] = (lat[:,-1]+self.depth)%3
                lat = np.roll(lat, -step[n], axis=n)
        return lat

    def generate_basis_element(self, state, step):      #adds only connected and new states to basis and bin_basis

        lat = self.make_step(state['lat'], step)
        seq = state['seq'] + [step]
        state = {'lat': lat, 'seq': seq}      #need to figure out if I want to save the state like this or as an array of arrays.
        # print(lat)
        # print(seq)

        if self.only_connected:
            physical = self.connected(state)
        else:
            physical = True
        new = False
        if physical:
            new = self.basis.add(self.state_2_list_entry(state) + [self.basis.length])
        check = new and physical
        if check:
            self.bin_basis.append(state)  #maybe better to put states in bins, divided into their sublattices
         
    def make_step(self, lat, step):  #done
        # lat = 2D boolean array with spin configurations (holes are False)
        # step = [x, y] gives the hopping
        x = np.ones((2,), dtype=int)* (self.depth +1) # position of the hole
        y = x + step #new hole position
        lat[x[0], x[1]],lat[(y)[0], (y)[1]] = lat[(y)[0], (y)[1]], lat[x[0], x[1]]


        # mark hole site as zero
        lat[y[0], y[1]] = 0
        #print(f'lat before translation: {lat}')
        lat = self.translation(lat, step)

        return lat

    def generate_basis(self):                                                       #under construction
        """ method to generate the entire basis (as list of 1D list of uint32) """
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
        print(f'generate 1hole basis for {lattice} lattice with {uc}')
        if self.basis.length > 0:
            print('Basis has already been built')
        else:
            lat = self.Neel_state[0]
            seq = []
            #Neel state
            state0 = {'lat': lat, 'seq': seq}
            #print(state0)

            self.basis.add(self.state_2_list_entry(state0) + [self.basis.length])
            self.bin_basis.append(state0)

            l = 0
            n0 = 0
            while l < self.depth:
                n1 = self.basis.length
                for n in range(n0, n1):
                    state0 = self.bin_basis[n]
                    seq = state0['seq']
                    for move in self.moves:
                        if self.honeycomb == True:
                            a = self.hole_is_on_2_sublattice({'seq': state0['seq']+[move]})
                            if a == False:
                                state_initial = copy.deepcopy(state0)
                                self.generate_basis_element(state_initial, np.array(move, dtype=int))
                        else:
                            state_initial = copy.deepcopy(state0)
                            self.generate_basis_element(state_initial, np.array(move, dtype=int))
                l += 1
                n0 = n1 

    def order_basis(self):
        #orders the bin basis
        ordered_list = []
        # ordered_list2 = []
        for x in self.basis.list:
            m = x[-1]
            ordered_list.append(self.bin_basis[m])
            # ordered_list2.append(self.bin_basis2[m])
        self.bin_basis = ordered_list
        # self.bin_basis = ordered_list2

    def matrix_el(self):    #should I include j_z & j_perp?
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
        self.col_V = []   
        self.row_V = []
        self.data_V = []
        self.data_j_perp = []
        self.col_j_perp = []
        self.row_j_perp = []

        steps = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
        steps2 = [np.array([1, 2]), np.array([2, 1]), np.array([1, -1])]  


        for i, state in enumerate(self.bin_basis):
        # for i in [0]:
        #     state = self.bin_basis[i]
        #     print(f'state {i}')
            lat = state['lat']
            seq = state['seq']
            sublattice = self.find_sublattice(state)
            lat0 = (lat-sublattice)%3
            diag = 0

            ####### compute diagonal part of H_J            seems to work
            lat_x = lat.copy()
            lat_x = self.translation(lat_x, [1,0])
            lat_y = lat.copy()
            lat_y = self.translation(lat_y, [0,1])
            lat_diag = lat.copy()
            lat_diag = self.translation(lat_diag, [1,1])

            jx_sum = np.count_nonzero(lat - lat_x ==0)
            jy_sum = np.count_nonzero(lat - lat_y ==0)
            jdiag_sum = np.count_nonzero(lat - lat_diag ==0)
            diag = (jx_sum + jy_sum + jdiag_sum)/2

            # remove contributions from links adjacent to one of the holes
            x = np.ones((2,), dtype=int)* (self.depth +1) 
            for move in self.moves:
                y = x + move
                if lat[x[0],x[1]] == lat[y[0],y[1]]:
                    diag -= 1/2

            self.data_j.append(diag)
            self.row_j.append(i)
            self.col_j.append(i)

            #### compute off-diagonal part of H_J          
            #### only allows spin flips that shorten the string. For string to be connected after flip only end
            siteslist = (np.argwhere(lat0)).tolist() # make sitelist either by retracing the seq or by subtracting neel background + argwhere
            for site in siteslist:
                nx=[[1,0],[0,1],[1,1]]
                for nn in nx:
                    if [site[0]+nn[0],site[1]+nn[1]] in siteslist:
                        lat1 = lat.copy()
                        lat1[site[0],site[1]],lat1[site[0]+nn[0],site[1]+nn[1]]=lat1[site[0]+nn[0],site[1]+nn[1]],lat1[site[0],site[1]]
                        state1 = {'lat': lat1, 'seq': seq}
                        a = self.state_2_list_entry(state1)
                        found, j  = self.basis.search(a)
                        if found:
                            self.data_j_perp.append(1)      
                            self.row_j_perp.append(j)
                            self.col_j_perp.append(i)


            #### compute H_{t}(k) part: 
            
            for step in steps:
                if np.all(np.abs(np.sum(seq+[step],axis=0))<self.depth+1):
                    lat1 = lat.copy()
                    lat1 = self.make_step(lat1, step)
                    state1 = {'lat': lat1, 'seq': seq + [step]}
                    # now state = H_{t}|i>
                    a = self.state_2_list_entry(state1)
                    found, j = self.basis.search(a)
                    if found:
                        self.row_t.append(j)
                        self.col_t.append(i)
                        self.data_t.append((-1*self.dist_2_phys_dist(step, seq)))
                        # print(f'phase: {-1*self.dist_2_phys_dist(step, seq)}, step: {step}')
                        # print(f'state {i} coupled to state {j}')
                        # print()
                

            ##### compute H_{t2}(k) part: 
            for step in steps2:
                lat1 = lat.copy()
                if np.all(np.abs(np.sum(seq+[step],axis=0))<self.depth+1):
                    lat1 = self.make_step(lat1, step)
                    state1 = {'lat': lat1, 'seq': seq + [step]}
                    # now state = H_{t'}|i>
                    a = self.state_2_list_entry(state1)
                    found, j = self.basis.search(a)
                    if found:
                        self.row_t2.append(j)
                        self.col_t2.append(i)
                        self.data_t2.append((-1*self.dist_2_phys_dist(step,seq)))

        self.data_t = np.array(self.data_t)
        self.data_t2 = np.array(self.data_t2)
        self.data_j = np.array(self.data_j)
        self.data_j_perp = np.array(self.data_j_perp)

    def compute_H(self, k, t=1.0, j=0.3, j_perp=0.3, t2=0):
    # uses list of data points from matrix_el_j and momentum to create sparse matrix H
    # k (array of size (2,1)) = hole momentum in LLP-frame
        if len(self.data_t) > 0:
            data_t = t * np.exp(1j * (self.data_t[:,0]*k[0]+self.data_t[:,1]*k[1]))
            row_t = self.row_t
            col_t = self.col_t
        else:
            data_t = []
            row_t = []
            col_t = []

        if bool(t2) and len(self.data_t2) > 0:
            data_t2 = t2 * np.exp(1j * (self.data_t2[:,0]*k[0]+self.data_t2[:,1]*k[1]))            
            row_t2 = self.row_t2
            col_t2 = self.col_t2
        else:
            data_t2 = []
            row_t2 = []
            col_t2 = []

        if bool(np.all(self.data_j_perp)):
            #data_j_perp = 1/2 * p ** data_1 * np.exp(1j * np.einsum('k,nk->n', k, data_2))
            data_j_perp = 1/2 * j_perp * self.data_j_perp
            row_j_perp = self.row_j_perp
            col_j_perp = self.col_j_perp
        else:
            data_j_perp = []
            row_j_perp = []
            col_j_perp = []

        self.data_t_test = data_t
        # print(f'tri data_t: {data_t}')
        # from collections import Counter
        # print('tri dictionary:',dict(Counter(sorted(data_t))))


        N = 1 #Normalization to Gellmann matrices
        # if self.honeycomb == False:
        data = np.concatenate((data_t, np.conj(data_t), data_t2, np.conj(data_t2), N*j * np.array(self.data_j), N * np.array(data_j_perp), N * np.conj(np.array(data_j_perp))), axis=0)
        row = np.array(row_t + col_t + row_t2 + col_t2 + self.row_j + row_j_perp + col_j_perp)
        col = np.array(col_t + row_t + col_t2 + row_t2 + self.col_j + col_j_perp + row_j_perp)
        

        self.data = data

        self.H = csr_matrix((data, (row,col)), shape=(len(self.bin_basis), len(self.bin_basis)), dtype=np.csingle)
        self.H.eliminate_zeros() # (only helpful if either t or j = 0)

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
        #Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
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
            return Es[state], vs[:,state] #default: returns the smallest eigenvalue, eigenstate

    def dispersion(self, k_array, state=0, t=1, t2=0, j=0.3, j_perp=0.3, two_D=False):
    #returns array of energies corresponding to the moments in k_array
    #2D == False: k_array = array of shape (Num_points,2)
    #2D == True: k_array = Meshgrid(x,y)
        if two_D:
            Ev = []
            k_x=k_array[0]
            k_y=k_array[1]
            E=np.empty(k_x.shape)
            #print(f'2d dispersion for band: {state}')
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t=t, t2=t2, j=j, j_perp=j_perp) 
                    E[i,l]=self.eigenval(state)
            
        else:
            E = [] 
            Ev = []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t=t, t2=t2, j=j, j_perp=j_perp)
                es, vs = self.eigensys(state)
                E.append(es)
                Ev.append(vs)
        return np.array(E), np.array(Ev)
    
    def dispersion_nmax(self,k_array,two_D=False, num_n=1, t=1, t2=0, j=0.3, j_perp=0.3):
    # returns array of energies corresponding to the moments in k_array
    # 2D == False: k_array = array of shape (Num_points,2)
    # 2D == True: k_array = Meshgrid(x,y)
        if two_D:
            k_x=k_array[0]
            k_y=k_array[1]
            Es=np.empty(k_x.shape + (num_n,))
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t=t, t2=t2, j=j, j_perp=j_perp)
                    E, _ = self.eigensys(num_n -1, full=True)
                    Es[i,l,:] = E
        else:
            E=[] 
            Evs= []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t=t, t2=t2, j=j, j_perp=j_perp)
                E.append(self.eigensys(num_n -1, full=True)[0])
                Evs.append(self.eigensys(num_n -1, full=True)[1])

        return np.array(E), np.array(Evs)
    
    def dispersion_nmax(self,k_array,two_D=False, num_n=1, t=1, t2=0, j=0.3, j_perp=0.3, p=-1, V=0):
        if two_D:
            k_x = k_array[0]
            k_y = k_array[1]
            # Initialize both containers matching the 2D grid dimensions
            Es = np.empty(k_x.shape + (num_n,))
            Evs = np.empty(k_x.shape + (num_n, len(self.bin_basis)), dtype=complex)
            
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l], k_y[i,l]], t=t, t2=t2, j=j, j_perp=j_perp)
                    E, V_sys = self.eigensys(num_n - 1, full=True)
                    Es[i,l,:] = E
                    Evs[i,l,:,:] = V_sys.T # Shape: (n_states, n_basis), stored to match grid
            return Es, Evs
        else:
            E=[] 
            Evs= []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t=t, t2=t2, j=j, j_perp=j_perp)
                E.append(self.eigensys(num_n -1, full=True)[0])
                Evs.append(self.eigensys(num_n -1, full=True)[1])
            return np.array(E), np.array(Evs)
        
    def rot_state_120(self, state):         #not tested, possible without for loop?
        lat = state['lat']
        seq = state['seq']

        sublattice = self.find_sublattice(state)
        lat1 = (lat-sublattice)%3 
        lat1[self.depth+1,self.depth+1] = 0
        row, col = np.nonzero(lat1)         #find all flipped sites
        val = lat[row,col]      
        rot_i = col - row + self.depth + 1 # New row index after rotation
        rot_j = -row + 2*(self.depth + 1)      # New column index after rotation
        #print(f'val{val},rot_i{rot_i},rot_j{rot_j}')
        rot_lat = sublattice.copy()
        rot_lat[rot_i, rot_j] = val

        #rotate  seq
        rot_seq = []
        for move in seq:
            move_rot = np.array([move[1]-move[0],-move[0]], dtype=int)
            rot_seq.append(move_rot)

        rot_state = {'lat': rot_lat, 'seq': rot_seq}
        return rot_state
    
    def build_rot_matrix(self,k):
        row = []
        col = []
        data = []

        for n, state in enumerate(self.bin_basis):
            seq = state['seq']
            rot_state = self.rot_state_120(state)
            found, m = self.basis.search(self.state_2_list_entry(rot_state))
            phase = 0
            if self.honeycomb and self.big_unit_cell:
                if len(state['seq'])%2 == 1:
                    # phase = k[0]*np.sqrt(3)
                    # phase = 1/2*(k[0] * np.sqrt(3) + k[1]*3)
                    phase = -1/2*(k[0] * np.sqrt(3) + k[1]*3)
            if not self.honeycomb and self.big_unit_cell:
                x = self.find_hole_sublattice(seq)
                if x == 1:
                    phase = -2*k[0]*np.sqrt(3)
                elif x == 2:
                    phase = -k[0]*np.sqrt(3)
            if found:
                row.append(m)
                col.append(n)
                data.append(np.exp(1j*phase))
            else:
                print('Couldnt find rotated state for rotation matrix')
        R = csr_matrix((data, (row, col)), shape=(self.basis.length, self.basis.length), dtype=np.csingle)
        return R
    
    def rot_trial_state_old(self, m3, k):
        lat = self.Neel_state[0]
        seq = []
        state0 = {'lat': lat, 'seq': seq}
        v = np.zeros((self.basis.length), dtype=complex)
        lat = np.zeros((self.L_size, self.L_size), dtype=bool)
        steps = [[-1, -1],[0, 1], [1, 0]] # green hole moves from sl 0 to 1
        for n, step in enumerate(steps):
            state = copy.deepcopy(state0)
            lat = self.make_step(state['lat'], step)
            seq = state['seq'] + [step]
            state = {'lat': lat, 'seq': seq}
            k_phase = 0
            if self.honeycomb and self.big_unit_cell:
                k_phase = k[0] * np.sqrt(3)

            a = self.state_2_list_entry(state)
            found, j = self.basis.search(a)
            
            if found:
                v[j] += 1/(np.sqrt(3)) * np.exp(1j * n * (2*np.pi/3 * m3)) * np.exp(-1j * k_phase) 
        return v
    
    def rot_trial_state(self, m3, k):
        lat = self.Neel_state[0]
        seq = []
        state0 = {'lat': lat, 'seq': seq}
        v = np.zeros((self.basis.length), dtype=complex)
        lat = np.zeros((self.L_size, self.L_size), dtype=bool)
        steps = [[-1, -1],[0, 1], [1, 0]] # green hole moves from sl 0 to 1
        
        # 1. Determine the gauge momentum phase per rotation
        k_phase = 0
        if self.honeycomb and self.big_unit_cell:
            # Matches the phase from build_rot_matrix for honeycomb odd sublattice
            k_phase = -1/2 * (k[0] * np.sqrt(3) + k[1] * 3)
        elif not self.honeycomb and self.big_unit_cell:
            # Matches the phase from build_rot_matrix for triangular sublattice 1
            k_phase = -2 * k[0] * np.sqrt(3)
            
        for n, step in enumerate(steps):
            state = copy.deepcopy(state0)
            lat = self.make_step(state['lat'], step)
            seq = state['seq'] + [step]
            state = {'lat': lat, 'seq': seq}

            a = self.state_2_list_entry(state)
            found, j = self.basis.search(a)
            
            if found:
                # 2. Multiply BOTH m3 and k_phase by n to apply the relative phase shifts
                v[j] += 1/(np.sqrt(3)) * np.exp(-1j * n * (2*np.pi/3 * m3 - k_phase)) 
        return v
# -----------------------------------------------------------------------------------

def run(args, Costum_k_path=False, Path=None):
    depth = args["depth"]
    t = args["t"]
    t2 = args["t2"]
    j = args["j"]
    j_perp = args["j_perp"]
    state = args["state"]
    connected = args["connected"]
    points_2D = args["points_2D"]
    grid_size = args["grid_size"]
    points_1D = args["points_1D"]
    honeycomb = args["honeycomb"]
    unit_cell = args["unit_cell"]
    D1 = args["1D_disp"]
    D2 = args["2D_disp"]
    honeycomb = args["honeycomb"]
    all_2D_bands = args["all_2D_bands"]
    Magnetic_BZ = args["Magnetic_BZ"]

    ### Create string basis
    t0 = perf_counter()
    sb = StringBasis(depth, connected, honeycomb, unit_cell)
    # print('coefficients:',depth, j, j_perp, t, t2, connected, state, grid_size, points_2D, points_1D, honeycomb, eunit_cell)
    print(f'coefficients:,depth={depth},j={j},j_perp={j_perp},t={t},t2={t2},connected={connected},state={state},grid_size={grid_size},points_2D={points_2D},points_1D={points_1D},honeycomb={honeycomb},unit_cell={unit_cell}')
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

    path_data = '/Users/linushein/Documents/Python/Python_output/SU(3)_truncated2'

    if D1 == True:
        if Costum_k_path:
            k_path=Path.T
        # print(f'k_grid shape:{k_path.shape}')
        # print(f'k_grid len:{len(k_path)}')
        t0 = perf_counter()
        E_1D = np.zeros([state,len(k_path)])
        for i in range(state):
            E_1D[i], _ = sb.dispersion(k_path, two_D=False, state=i, t=t, t2=t2, j=j, j_perp=j_perp)
            #print(f'band {i}: E = {E_1D[i]}')
        print('computed 1D dispersion in {t:.3f}s'.format(t=perf_counter()-t0))
        # print(f'E_1D shape: {E_1D.shape}')  
        # if Custom_k_path:
        #     E_1D=E_1D.T
        if honeycomb == True:
            name2 = f'string_1hole_1D_disp_SU2_Honeycomb_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands.npy'
        else:
            name2 = f'string_1hole_1D_disp_SU(3)_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands.npy'
        np.save(os.path.join(path_data, name2), E_1D)

    if D2 == True:
        t0 = perf_counter()
        x = copy.deepcopy(state)
        if all_2D_bands == False:
            x = 1
        E_2D = np.zeros([x,points_2D,points_2D])
        ###Calculate Dispersion
        for i in range(x):
            E_2D[i], _ = sb.dispersion(k_grid, t=t, t2=t2, j=j, j_perp=j_perp, two_D=True, state=i)
            print(f'for band {i}:')
            print('computed 2D dispersion in {t:.3f}s'.format(t=perf_counter()-t0))
        if honeycomb == True:
            name = f'string_1hole_2D_disp_SU2_Honeycomb_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands.npy'
        else:
            name = f'string_1hole_2D_disp_SU(3)_depth{depth}_t{t}_t2{t2}_J{j}_Jperp{j_perp}_{state}bands.npy'
        np.save(os.path.join(path_data, name), E_2D)


if __name__ == "__main__":
    args = {
        "depth": 3,
        "j": 0.30,
        "j_perp": 0.30,
        "t": 1.0,
        "t2": 0,
        "fermions": True,   #irrelevant for single hole
        "connected": True,  
        "state": 5,
        "grid_size": 2*np.pi,  
        "points_2D": 120,
        "points_1D": 60,
        "honeycomb": True,
        "unit_cell": 0,
        "1D_disp": True,
        "2D_disp": False,
        "all_2D_bands": True,
        "Magnetic_BZ": True,
    }
    run(args)



