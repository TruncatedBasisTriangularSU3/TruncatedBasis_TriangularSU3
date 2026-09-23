# import necessary libraries
import numpy as np
import copy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from time import perf_counter

def greater_arr(a, b, length):
### lexicographic comparison of lists
    return a[:length]>b[:length]

def brick_to_hc_distance(d):
        ''' converts distance dx, dy on brickwall lattice to distance r1, r2 on honeycomb lattice '''
        if len(d.shape)<2:
            d = d.reshape(1,d.shape)
        assert np.sum(d)%2 == 0 #check that d is already resized to same sl distance
        r1 = -d[:,0]
        r2 = (d[:,1]+d[:,0])//2
        return r1, r2

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
    def __init__(self, depth, only_connected=False):  # works only for depth<=14 otherwise change uint32 to uint64!
        self.depth = depth
        self.L_size = 2*self.depth+3
        self.basis = sorted_list(length_arr = self.L_size + 3)
        self.bin_basis=[]
        self.last_move = [None, None]
        self.moves = [[[0, -1, 0], [0, 0, -1], [0, 0, 1]],[[0,1,0],[0,0,-1],[0,0,1]]] + [[[1, -1, 0], [1, 0, -1], [1, 0, 1]], [[1,1,0],[1,0,-1],[1,0,1]]] #moves: hole 0 on sl 0, hole 0 on sl 1, hole1 on sl 0, hole1 on sl 1
        self.H=None
        self.col_t=[]
        self.row_t=[]
        self.data_t=None
        self.data_t=None
        self.col_j=[]
        self.row_j=[]
        self.data_j=[]
        self.representatives = [] #basis of representative states, only this is used in Hamiltonian (like Fock basis)
        self.is_representative = [] #array with 3 entries per state: bool (representative or not), position of state with exchanged holes in bin_basis, position of representative in representatives basis
        self.only_connected = only_connected
        self.rot_mat = None

        
        self.tol=0

        self.generate_basis()
        self.order_basis()
        self.generate_representatives()
        self.matrix_el()
        #self.calc_rot_mat()

    def connected(self, state):
        # state should be list of dictionary with keys 'lat', 'hole_pos', 'seq'
        res = False
        a = state['lat']
        hole_pos = state['hole_pos']
        # construct list of sites with flipped spins

        flipped = (np.argwhere(a)).tolist()
        flipped = flipped + [list(hole_pos[0])] + [list(hole_pos[1])]
        # determine if the flipped spins are connected
        if len(flipped) == 0:
            res = True
        else:
            moves = list(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]))
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

    def state_2_list_entry(self, state):
        lat = state['lat']
        hole_pos = state['hole_pos']
        sl = state['sl']
        a = lat*np.matmul(np.ones((self.L_size, 1), dtype=np.uint32),
                          np.reshape(2**np.arange(self.L_size, dtype=np.uint32), (1, self.L_size)), dtype=np.uint32)
        list_entry = np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos] +[sl]
        #print("list entry=", list_entry)
        return list_entry

    def generate_step(self, l, state0): #?????????
    # function which recursively generates all possible paths
        for move in self.moves:
            state_initial = copy.deepcopy(state0)
            physical, state = self.generate_basis_element(state_initial, np.array(move, dtype=int))
            if l > 1 and physical:
                self.generate_step(l-1, state)
           
    def generate_basis_element(self, state, step):
    # an old state and a step this function generetates a basis element and adds it to the basis list
    # state = dict(')
    # seq is a list of steps = np.arrays where seq[i] = [n, x, y]
    # n = 0,1 denotes if the first (0) or second (1) hole moves
    # x,y in {-1, 0, 1} denote the direction of the hopping
        lat, hole_pos, sl = self.make_step(state['lat'], state['hole_pos'], state['sl'], step)
        #sl = state['sl']
        #sl[step[0]] = (sl[step[0]]+1)%2 #sublattice of hole that hopped changes
        state = {'lat': lat, 'hole_pos': hole_pos, 'sl': sl}
        #print(state)
        # check if the two holes site on the same site
        assert (hole_pos[0]==self.depth + 1).all()
        if self.only_connected:
            physical = (not (hole_pos[0] == hole_pos[1]).all()) and self.connected(state)
        else:
            physical = not (hole_pos[0] == hole_pos[1]).all()
        new = False
        if physical:
            new = self.basis.add(self.state_2_list_entry(state) + [self.basis.length])
        check = new and physical
        #print(new)
        #print(check)
        if check:
            self.bin_basis.append(state)
    
    def translation(self, lat, hole_pos, step):
        for n in range(2):
            lat = np.roll(lat, -step[n + 1], axis=n)
            hole_pos[n] = hole_pos[n] - step[1:]
        return lat, hole_pos
    
    def make_step(self, lat, hole_pos, sl, step):
        # lat = 2D boolean array with spin configurations (holes are False)
        # step = [n, x, y] gives the hole and hopping
        # hole_pos = list of arrays [[x1, y1], [x2, y1]] gives the position of both holes
        n = step[0]
        x = hole_pos[n] #old hole position of moving hole
        y = hole_pos[n] + step[1:] #new hole position of moving hole
        sl[n] = (1-sl[n])
        #print("x,y=",x,y)
        if np.sum(np.abs(step[1:])) % 2 == 1: #hole jumps and uneven number of steps in total => spin at center opposite from new hole site (spin becomes True if hole hops on False site (spin flipped), becomes False if on True site (spin flipped back)
            lat[x[0], x[1]] = not(lat[y[0], y[1]])
        else:
            lat[x[0], x[1]] = lat[y[0], y[1]]
        hole_pos[n] = y #new hole position

        # mark hole site as False
        for n in range(2):
            [y, x] = hole_pos[n]
            lat[y, x] = False

        if step[0] == 0 : #central hole moved, translate lattice
            lat, hole_pos = self.translation(lat, hole_pos, step)
            
        #change sublattice of hole that moves
        return lat, hole_pos, sl

    def exchange_holes(self, state):
        lat = state['lat'].copy()
        hole_pos = state['hole_pos'].copy()
        sl = state['sl'].copy()
        delta = hole_pos[1].copy()
        hole_pos[1] = hole_pos[0].copy()
        hole_pos[0] = delta
        step = np.concatenate((np.zeros((1,), dtype=int), delta - self.depth - 1), axis=None)
        lat, hole_pos = self.translation(lat, hole_pos, step)
        state['lat'] = lat
        state['hole_pos'] = hole_pos
        state['sl'] = [sl[1],sl[0]] #sublattices exchanged
        return state

    def generate_basis(self):
    # method to generate the entire basis (as list of 1D list of uint32)
        lat = np.zeros((self.L_size, self.L_size), dtype=bool)
        hole_pos = [np.ones((2,), dtype=int)* (self.depth +1)] * 2 #both holes on central site
        sl = [0,0]
        state0 = {'lat': lat, 'hole_pos': hole_pos, 'sl': sl}
        #sl: [sublattice of first hole, sublattice of second hole]
        for h in range(2): #consider both holes
            for move in self.moves[2*h+sl[h]]: #move one hole to generate starting states
                #print(f"hole {h} moves {move[1:]}")
                state_initial = copy.deepcopy(state0)
                self.generate_basis_element(state_initial, np.array(move, dtype=int))
        l = 1
        n0 = 0
        while l < self.depth:
            n1 = self.basis.length
            for n in range(n0, n1):
                state0 = self.bin_basis[n]
                sl = state0['sl']
                for h in range(2):
                    for move in self.moves[2*h+sl[h]]:
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
        self.is_representative = [(False, 0)] * self.basis.length
        for n in range(self.basis.length):
            #print("old state", self.bin_basis[n])
            state = copy.deepcopy(self.bin_basis[n])
            #print(state)
            state = self.exchange_holes(state)
            #print(state)
            #print("new state", state)
            a = self.state_2_list_entry(state)
            found, m = self.basis.search(a)
            assert found
            if m >= n:
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
        # self.col_t2 = []
        # self.row_t2 = []
        # self.data_t2 = []
        self.col_j = []
        self.row_j = []
        self.data_j = []
        self.col_V = []
        self.row_V = []
        self.data_V = []
        self.data_j_perp = []
        self.col_j_perp = []
        self.row_j_perp = []
        steps = [np.array([1, 1, 0]), np.array([1, 0, 1])]
        #steps2 = [np.array([1, 1, 1]), np.array([1, 1, -1])]
        #print("computing matrix elements")
        for i in range(len(self.representatives)):
            state = self.representatives[i]
            #print("state=",state)
            lat = state['lat']
            hole_pos = state['hole_pos']
            sl = state['sl']

            ####### compute density density term, 
            #print(np.abs(hole_pos[0]-hole_pos[1]))
            if np.all(np.abs(hole_pos[0]-hole_pos[1])==[0,1]) or np.all(np.abs(hole_pos[0]-hole_pos[1])==[(-1)**sl[0],0]):
                #print("holes adjacent")
                #print(np.abs(hole_pos[0]-hole_pos[1]))
                diag = 0.5 * 7
                self.data_V.append(1)
                self.row_V.append(i)
                self.col_V.append(i)
            else:
                diag = 0.5 * 8

            ####### compute H_{J_z}
            #bond to the top: even sites if hole on sublattice A (sl=0), odd if B (sl=1)
            latsum_up = np.logical_xor(lat, np.roll(lat, -1, axis=0)).flatten()
            latsum_up[(sl[0])::2]=0 #set indices with no upper connection to zero
            diag = np.sum(latsum_up)
            #print(diag)
            #horizontal conectivity as for square
            diag += np.sum(np.logical_xor(lat, np.roll(lat, 1, axis=1)))
            #print(diag)
            # remove contributions from links adjacent to one of the holes
            neighbours = []
            for n in range(2):
                xh = hole_pos[n][1]
                yh = hole_pos[n][0]
                #print('hole pos', yh, xh)
                for x in [-1, 1]:
                    neighbours.append([yh, xh+x])
                    # print('checking', yh, x+xh)
                    # if lat[yh, x+xh]:
                    #     diag -= 1
                    # print(n, 'hole x', diag)
                if sl[n]==0:
                    y = -1
                    neighbours.append([yh+y, xh])
                    # print('checking', yh+y, xh)
                    # if lat[yh+y, xh]:
                    #     diag -= 1
                elif sl[n]==1:
                    y = 1
                    neighbours.append([yh+y, xh])
                #     print('checking', yh+y, xh)
                #     if lat[yh+y, xh]:
                #         diag -= 1
                # print(n, 'hole y', diag)
            #print(neighbours)
            for nei in neighbours:
                if lat[nei[0], nei[1]]:
                    diag -= 1
            #print(diag)

            self.data_j.append(diag*1/2)
            self.row_j.append(i)
            self.col_j.append(i)

            ##### compute H_{J_perp} part
            #r = np.array([1, 1])*(1+self.depth)-np.sum(seq, axis=0)[1:] #what does this do?

            siteslist = list(np.argwhere(lat))

            for site in siteslist:
                lat1=lat.copy()
                #bonds along x
                if lat1[site[0],site[1]] and lat1[site[0],site[1]+1]:
                    lat1[site[0],site[1]]=False
                    lat1[site[0],site[1]+1]=False
                    a = lat1*np.matmul(np.ones((self.L_size, 1), dtype=np.uint32), np.reshape(2**np.arange(self.L_size, dtype=np.uint32), (1, self.L_size)), dtype=np.uint32)
                    #a = self.state_2_list_entry({'lat': lat1.copy()})
                    found,m = self.basis.search(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos]+[sl])
                    rep, _, j = self.is_representative[m]
                    if found:
                        self.row_j_perp.append(j)
                        self.row_j_perp.append(i)
                        self.col_j_perp.append(i)
                        self.col_j_perp.append(j)
                        if rep:
                            self.data_j_perp.append((0,np.zeros((2,), dtype=int), sl))
                            self.data_j_perp.append((0,np.zeros((2,), dtype=int), sl))
                        else:
                            # same hole-hole displacement convention as calc_rot_mat:
                            # shift to a same-sublattice distance, otherwise the //2 in
                            # brick_to_hc_distance floor-divides an odd number.
                            phase = (hole_pos[1] - self.depth - 1)
                            if sl[0]!=sl[1]:
                                phase[0] += 2*(sl[0]-0.5)
                            self.data_j_perp.append((1,-1*phase, sl))
                            self.data_j_perp.append((1,phase.copy(), sl))

                    # restore the two sites flipped above (site[0],site[1]+1), not (site[0]+1,site[1])
                    lat1[site[0],site[1]]=True
                    lat1[site[0],site[1]+1]=True

                if (self.L_size*site[0]+site[1]+sl[0])%2==1: #bond downwards
                    if lat1[site[0],site[1]] and lat1[site[0]+1,site[1]]:
                        lat1[site[0],site[1]]=False
                        lat1[site[0]+1,site[1]]=False
                        a = lat1*np.matmul(np.ones((self.L_size, 1), dtype=np.uint32), np.reshape(2**np.arange(self.L_size, dtype=np.uint32), (1, self.L_size)), dtype=np.uint32)
                        #a = self.state_2_list_entry({'lat': lat1.copy()})
                        found,m=self.basis.search(np.sum(a, axis=1).tolist() + [list(pos) for pos in hole_pos]+[sl])
                        rep, _, j = self.is_representative[m]
                        if found:
                            self.row_j_perp.append(j)
                            self.row_j_perp.append(i)
                            self.col_j_perp.append(i)
                            self.col_j_perp.append(j)
                            if rep:
                                self.data_j_perp.append((0,np.zeros((2,), dtype=int), sl))
                                self.data_j_perp.append((0,np.zeros((2,), dtype=int), sl))
                            else:
                                phase = (hole_pos[1] - self.depth - 1)
                                if sl[0]!=sl[1]:
                                    phase[0] += 2*(sl[0]-0.5)
                                self.data_j_perp.append((1,-1*phase, sl))
                                self.data_j_perp.append((1,phase, sl))

            ##### compute H_{t}(k) part: #hole 1 hops
            steps = [np.array([1,0,1]), np.array([1,0,-1]), np.array([1,1,0])]
            #print("state=", state)
            if sl[1]==1:
                for step in steps:
                    lat1 = lat.copy()
                    sl1 = sl.copy()
                    hole_pos1 = hole_pos.copy()
                    #print(lat1, sl1, hole_pos1)
                    #print("step=", step)
                    lat_new, hole_pos_new, sl_new = self.make_step(lat1, hole_pos1, sl1, step)
                
                    state_new = {'lat': lat_new, 'hole_pos': hole_pos_new, 'sl': sl_new}
                    #print("new state", state_new)
                    # now state = H_{t}|i>
        
                    a = self.state_2_list_entry(state_new)
                    found, m = self.basis.search(a)
                    #print(found)
                    if found:
                        (rep, _, m3) = self.is_representative[m]
                        # print('found ', i, m3, rep, step)
                        self.row_t.append(m3)
                        self.col_t.append(i)
                        if rep:
                            #print("rep phase", 0)
                            self.data_t.append((0, np.zeros((2,), dtype=int), sl_new))
                        else:
                            #print("no rep phase", (hole_pos_new[1] - self.depth - 1))
                            phase = (hole_pos_new[1] - self.depth - 1)
                            if sl_new[0]!=sl_new[1]:
                                phase[0]+= 2*(sl_new[0]-0.5)
                            self.data_t.append((1, phase, sl_new))
                        # print(self.data_t[-1])
            state1 = copy.deepcopy(state)
            sl0 = state1['sl']
            #print("0", sl0)
            #print("state", state)
            la = state1['hole_pos'][1] - self.depth - 1
            if sl[0]!=sl[1]:
                la[0]+= 2*(sl[0]-0.5)
            state_ex = self.exchange_holes(state1)  
            sl_ex = state_ex['sl']
            lat_ex = state_ex['lat']
            hole_pos_ex = state_ex['hole_pos']
            #print("ex", sl_ex)
            if sl_ex[1]==1: 
                #print("state exchanged", state_ex)
                for step in steps:
                    lat1 = lat_ex.copy()
                    hole_pos1 = hole_pos_ex.copy()
                    sl1 = sl_ex.copy()
                    #print("1", sl1)
                    
                    #print("step=", step)
                    #print(lat1, hole_pos1, sl1)
                    #print("state next step", lat_ex, hole_pos_ex)
                    lat_new, hole_pos_new, sl_new = self.make_step(lat1, hole_pos1, sl1, step)
                    state_new = {'lat': lat_new, 'hole_pos': hole_pos_new, 'sl': sl_new}
                    #print("new state", state_new)
                    # now state = H_{t}|i>
                    a = self.state_2_list_entry(state_new)
                    found, m = self.basis.search(a)
                    #print(found)
                    #print("1,new", sl1, sl_new)
                    if found:
                        (rep, _, m3) = self.is_representative[m]
                        # print('found2', i, m3, rep, step)
                        # print(state1['hole_pos'])
                        self.row_t.append(m3)
                        self.col_t.append(i)
                        if rep:
                            #print("rep phase", la)
                            self.data_t.append((1, la, sl_new))
                        else:
                            #if sl_ex[0]==sl_ex[1]:
                                #la[1] += 1
                            #print("no rep phase", (hole_pos_new[1] - self.depth - 1 + la))
                            phase = (hole_pos_new[1] - self.depth - 1 + la)
                            if sl_new[0]!=sl_new[1]:
                                phase[0]+=2*(sl[0]-0.5)
                            self.data_t.append((0, phase, sl_new))
                        #print(self.data_t[-1])
                        #print(la, hole_pos1[1] - self.depth -1)

            ##### compute H_{t'}(k) part:
            # for step in steps2:
            #     lat1 = lat.copy()
            #     hole_pos1 = copy.deepcopy(hole_pos)
            #     lat1, hole_pos1 = self.make_step(lat1, hole_pos1, step)
            #     state1 = {'lat': lat1, 'hole_pos': hole_pos1}
            #     # now state = H_{t'}|i>
            #     a = self.state_2_list_entry(state1)
            #     found, m = self.basis.search(a)
            #     if found:
            #         (rep, _, m3) = self.is_representative[m]
            #         self.row_t2.append(m3)
            #         self.col_t2.append(i)
            #         if rep:
            #             self.data_t2.append((0, np.zeros((2,), dtype=int)))
            #         else:
            #             self.data_t2.append((1, -1*(hole_pos1[1] - self.depth - 1)))

            #     state1 = copy.deepcopy(state)
            #     la = state['hole_pos'][1] - self.depth - 1
            #     state1 = self.exchange_holes(state1)   
            #     lat1 = state1['lat']
            #     hole_pos1 = state1['hole_pos']
            #     lat1, hole_pos1 = self.make_step(lat1, hole_pos1, step)
            #     state1 = {'lat': lat1, 'hole_pos': hole_pos1}
            #     # now state = H_{t'}|i>
            #     a = self.state_2_list_entry(state1)
            #     found, m = self.basis.search(a)
            #     if found:
            #         (rep, _, m3) = self.is_representative[m]
            #         self.row_t2.append(m3)
            #         self.col_t2.append(i)
            #         if rep:
            #             self.data_t2.append((1, -1 * la))
            #         else:
            #             self.data_t2.append((0, -1 * (hole_pos1[1] - self.depth - 1 + la)))
    
    def eigenval(self, state=0):
    # computes smallest eigenvalue of H
        Es, _ = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
        return np.sort(Es)[state]
    
    def eigenvec(self, state=0, v0=False):
    #computes eigenvector corresponding to the smallest eigenvalue
        if len(self.representatives) == 3:
            Es, vs = eigh(self.H.toarray())
            # if Es[0] == Es[1]:
            #     return np.array([1, -1]) / np.sqrt(2)
            # else:
            return vs[:, state]
        else:
            if v0:
                v0 = np.ones((len(self.representatives),))/np.sqrt(len(self.representatives))
                Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol, v0=v0)
            else:
                Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
            sort_ind = np.argsort(Es)
            vs = vs[:, sort_ind]
            return vs[:,state]
    
    def eigensys(self, state=0, full=False, dense=False):
    # computes smallest eigenvalue of H
    # dense=True: full double-precision diagonalisation. H is stored in single
    # precision, and ARPACK returns only ~1e-3 orthonormal vectors inside a
    # near-degenerate multiplet, which makes any per-band decomposition noisy.
        if dense:
            Es, vs = np.linalg.eigh(self.H.toarray().astype(np.complex128))
            Es = Es[:state+1]
            vs = vs[:,:state+1]
        else:
            Es, vs = eigsh(self.H,k=state+1,which='SA',tol=self.tol)
        sort_ind = np.argsort(Es)
        Es = Es[sort_ind]
        vs = vs[:, sort_ind]
        if full:
            return Es, vs
        else:
            
            return Es[state], vs[:,state]
    
    def brick_to_hc_distance(self, d):
        ''' converts distance dx, dy on brickwall lattice to distance r1, r2 on honeycomb lattice '''
        assert np.sum(d)%2 == 0 #check that d is already resized to same sl distance
        r1 = d[0]
        r2 = (d[1]-d[0])//2
        return r1, r2
        
    def compute_H(self,k, t, j, j_perp, t2=0, p=-1, V=0):
    # uses list of data points from matrix_el_j and momentum to create sparse matrix H
    # k (array of size (2,1)) = hole momentum in LLP-frame
        
        if len(self.data_t) > 0:
            data_1 = np.array([x[0] for x in self.data_t])
            data_2 = np.array([x[1] for x in self.data_t])
            data_3 = np.array([x[2] for x in self.data_t])
            #sign = 2*(data_3[:,1]-0.5)
            #r2 = data_2[:,0]+sign*(data_3[:,0]+data_3[:,1])%2
            #r1 = (data_2[:,1]-r2)//2
            #r1=data_2[:,0]
            #r2=(data_2[:,1]-data_2[:,0])//2
            r1, r2 = brick_to_hc_distance(data_2)
            rx = np.sqrt(3)*r2+np.sqrt(3)/2*r1
            ry = 3/2*r1
            #rx = np.sqrt(3)/2*(data_2[:,1])
            #ry = -3/2*(data_2[:,0])+1/2*np.mod(np.sum(data_2, axis=1), 2)
            #print("ry shape", ry.shape)
            #ind = np.where(data_3[:,0]!=data_3[:,1])[0]
            #print(np.where(data_3[:,0]!=data_3[:,1]))
            #print("ind shape", ind.shape)
            #ry[ind] = ry[ind] + 2*(data_3[ind,1]-0.5)

            data_t = t * p ** data_1 * np.exp(-1j * (k[0]*rx+k[1]*ry))
            #data_t = t * (p ** data_1) * np.exp(-1j * (np.sqrt(3)*k[0]*(r2/2+r1)+3/2*k[1]*r1))
        else:
            data_t = []

        # if bool(t2) and len(self.data_t2) > 0: #this doesn't work yet
        #     data_1 = np.array([x[0] for x in self.data_t2])
        #     data_2 = np.array([x[1] for x in self.data_t2])
        #     data_t2 = t2 * (p ** data_1) * np.exp(1j * np.einsum('k,nk->n', k, data_2))
        #     row_t2 = self.row_t2
        #     col_t2 = self.col_t2
        # else:
        data_t2 = []
        row_t2 = []
        col_t2 = []

        if bool(self.data_j_perp):
            data_1 = np.array([x[0] for x in self.data_j_perp])
            data_2 = np.array([x[1] for x in self.data_j_perp])
            data_3 = np.array([x[2] for x in self.data_t])
            #sign = 2*(data_3[:,1]-0.5)
            #r2 = data_2[:,0]+sign*np.sum(data_2, axis=1)%2
            #r1 = (data_2[:,1]-r2)//2
            # rx = np.sqrt(3)/2*(data_2[:,1])
            # #print(data_2)
            # #print(data_2.shape)
            # ry = -3/2*(data_2[:,0])+1/2*np.mod(np.sum(data_2, axis=1), 2)
            # r1=data_2[:,0]
            # r2=(data_2[:,1]-data_2[:,0])//2
            # rx = np.sqrt(3)*r2+np.sqrt(3)/2*r1
            # ry = -3/2*r1
            r1, r2 = brick_to_hc_distance(data_2)
            rx = np.sqrt(3)*r2+np.sqrt(3)/2*r1
            ry = 3/2*r1
            #ind = np.where(data_3[:,0]!=data_3[:,1])[0]
            #print(np.where(data_3[:,0]!=data_3[:,1]))
            #print("ind shape", ind.shape)
            #ry[ind] = ry[ind] - 2*(data_3[ind,1]-0.5)
            #print(data_2)
            #print(rx, ry)
            #print(rx*k[0]+ry*k[1])
            data_j_perp = 1/2 * p ** data_1 * np.exp(-1j * (k[0]*rx+k[1]*ry))
            #data_j_perp = 1/2 * p ** data_1 * np.exp(-1j * (np.sqrt(3)*k[0]*(r2/2+r1)+3/2*k[1]*r1))
            row_j_perp = self.row_j_perp
            col_j_perp = self.col_j_perp
        else:
            data_j_perp = []
            row_j_perp = []
            col_j_perp = []
        data = np.concatenate((data_t, np.conj(data_t), data_t2, np.conj(data_t2), j * np.array(self.data_j), j_perp * np.array(data_j_perp), V * np.array(self.data_V)), axis=0)
        row = np.array(self.row_t + self.col_t + row_t2 + col_t2 + self.row_j + row_j_perp + self.row_V)
        col = np.array(self.col_t + self.row_t + col_t2 + row_t2 + self.col_j + col_j_perp + self.col_V)

        self.H = csr_matrix((data, (row,col)), shape=(len(self.representatives), len(self.representatives)), dtype=np.csingle)
        self.H.eliminate_zeros() # (only helpful if either t or j = 0)
    
    def rot_trial_state(self, m3, k, p=-1):
        '''Computes unnormalized trial state with given rotational C3 eigenvalue m6 and momentum k'''
        assert len(self.representatives) > 0
        v = np.zeros((len(self.representatives)), dtype=complex)
        lat = np.zeros((self.L_size, self.L_size), dtype=bool)
        hole_pos = [np.ones((2,), dtype=int)* (self.depth +1)] * 2
        sl = [0,0]
        state0 = {'lat': lat, 'hole_pos': hole_pos, 'sl': sl}
        steps = [[1, 0, 1], [1, 0, -1], [1, -1, 0]] #hole 0 moves from sl 0 to 1
        for n, step in enumerate(steps):
            step = np.array(step, dtype=int)
            state = copy.deepcopy(state0)
            lat, hole_pos, sl = self.make_step(state['lat'], state['hole_pos'], state['sl'], step)
            state = {'lat': lat, 'hole_pos': hole_pos, 'sl': sl}

            # search for this state in the basis of representatives
            found,m=self.basis.search(self.state_2_list_entry(state))
            rep, _, j = self.is_representative[m]
            if found:
                phase = 0.
                if not rep:
                    d2 = (hole_pos[1] - self.depth - 1)
                    #print(d2)
                    if sl[0]!=sl[1]:
                        d2[0] += 2*(sl[0]-0.5)
                    r1=d2[0]
                    r2=(d2[1]-d2[0])//2
                    rx = np.sqrt(3)*r2+np.sqrt(3)/2*r1
                    ry = -3/2*r1
                    phase = np.pi + k[0]*rx+k[1]*ry
                if not sl[0]==0:
                    phase += 1/2*(np.sqrt(3)*k[0]+3*k[1])
                v[j] += 1/(np.sqrt(3)) * np.exp(1j * m3 * n * 2*np.pi/3 + 1j*phase)
        return v
    
    def rot_trial_state_from_rep(self, m3, k, p=-1):
        '''Computes unnormalized trial state with given rotational C3 eigenvalue m6 and momentum k'''
        assert len(self.representatives) > 0
        v = np.zeros((len(self.representatives)), dtype=complex)
        reps = self.representatives
        for i, state in enumerate(reps):
            lat = state['lat']
            sl = state['sl']
            #print(lat, sl)
            #print(state['hole_pos'])
            length = np.sum(lat)
            if length ==0:
                v[i] = 1/(np.sqrt(3))
                if sl!=0:
                    phase_rot = -1/2*(np.sqrt(3)*k[0]+3*k[1])
                else:
                    phase_rot = 0
                #rotate state once
                rot_state_1 = self.rotate_state(state)
                rot_state_2 = self.rotate_state(self.rotate_state(state))


                found, m1 = self.basis.search(self.state_2_list_entry(rot_state_1))
                if not found:
                    raise("Wrong Rotation, state not in basis!")
                rep1, _, j1 = self.is_representative[m1]
                if not rep1:
                    hole_pos_1 = rot_state_1['hole_pos']
                    sl_1 = rot_state_1['sl']
                    d2 = (hole_pos_1[1] - self.depth - 1)
                    if sl_1[0]!=sl_1[1]:
                        d2[0] += 2*(sl[0]-0.5)
                    r1=d2[0]
                    r2=(d2[1]-d2[0])//2
                    rx = np.sqrt(3)*r2+np.sqrt(3)/2*r1
                    ry = -3/2*r1
                    phase_ex_1 = np.pi+k[0]*rx+k[1]*ry
                else:
                    phase_ex_1 = 0
                v[j1] +=  1/(np.sqrt(3)) *  np.exp(1j * (m3 * 2*np.pi/3)-1j*phase_ex_1)
                
                #rotate state twice
                found, m2 = self.basis.search(self.state_2_list_entry(rot_state_2))
                rep2, _, j2 = self.is_representative[m2]
                if not rep2:
                    hole_pos_2 = rot_state_2['hole_pos']
                    sl_2 = rot_state_2['sl']
                    d2 = (hole_pos_2[1] - self.depth - 1)
                    if sl_2[0]!=sl_2[1]:
                        d2[0] += 2*(sl[0]-0.5)
                    r1=d2[0]
                    r2=(d2[1]-d2[0])//2
                    rx = np.sqrt(3)*r2+np.sqrt(3)/2*r1
                    ry = -3/2*r1
                    phase_ex_2 = np.pi+k[0]*rx+k[1]*ry
                else:
                    phase_ex_2 = 0
                #v[j2] += 1/(np.sqrt(3)) * np.exp(1j * m3 * 2 * 2*np.pi/3 - 1j*phase)
                v[j2] = 1/np.sqrt(3)*np.exp(1j*2*(m3 * 2*np.pi/3)-1j*phase_ex_2)
            break
        # for j in range(len(self.representatives)):
        #     if v[j] != 0:
        #         print(j, np.angle(v[j])/np.pi, np.abs(v[j]))
        return v

    def delete_weak_j_perp(self, t, jz, j_perp, cutoff=1e-4):
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
            self.compute_H(k, t, jz, j_perp, p=-1)
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

    def dispersion(self,k_array,two_D=False, state=0, t=1, j=0.3, j_perp=0.3, p=-1, V=0):
    # returns array of energies corresponding to the moments in k_array
    # 2D == False: k_array = array of shape (Num_points,2)
    # 2D == True: k_array = Meshgrid(x,y)
        if two_D:
            k_x=k_array[0]
            k_y=k_array[1]
            E=np.empty(k_x.shape)
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t, j, j_perp, p=p) 
                    E[i,l]=self.eigenval(state)

        else:
            E=[] 
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t, j, j_perp, p=p, V=V)
                E.append(self.eigenval(state))
        return np.array(E)

    def dispersion_nmax(self,k_array,two_D=False, num_n=1, t=1, j=0.3, j_perp=0.3, p=-1, V=0, dense=False):
    # returns array of energies corresponding to the moments in k_array
    # 2D == False: k_array = array of shape (Num_points,2)
    # 2D == True: k_array = Meshgrid(x,y)
        if two_D:
            k_x=k_array[0]
            k_y=k_array[1]
            Es=np.empty(k_x.shape + (num_n,))
            for i in range(k_x.shape[0]):
                for l in range(k_x.shape[1]):
                    self.compute_H([k_x[i,l],k_y[i,l]], t, j, j_perp, p=p)
                    E, _ = self.eigensys(num_n -1, full=True)
                    Es[i,l,:] = E
        else:
            E=[]
            evs = []
            for i in range(k_array.shape[0]):
                k=k_array[i,:]
                self.compute_H(k, t, j, j_perp, p=p, V=V)
                energies, vectors = self.eigensys(num_n -1, full=True, dense=dense)
                E.append(energies)
                evs.append(vectors)
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
            n1, n2, n3 = self.brick_to_hc(x,y,sl[0])
            #print(n1,n2,n3)
            new_x, new_y =  self.hc_to_brick(n3,n1,n2,sl[0])
            #print(new_x, new_y)
            new_lat[new_y+depth+1,(new_x+depth+1)] = True
        x_hole_1 = state['hole_pos'][1][1]-depth-1
        y_hole_1 = state['hole_pos'][1][0]-depth-1
        n1, n2, n3 = self.brick_to_hc(x_hole_1,y_hole_1,sl[0])
        new_x_hole_1, new_y_hole_1 = self.hc_to_brick(n3,n1,n2,sl[0])
        new_hole_pos = [state['hole_pos'][0], np.array([new_y_hole_1+depth+1, new_x_hole_1+depth+1], dtype=int)]
        new_state = {'lat': new_lat, 'sl': sl, 'hole_pos': new_hole_pos}
        return new_state
    
    def calc_rot_mat(self, k):
        Rm = np.zeros((len(self.representatives), len(self.representatives)), dtype=complex)
        R_phys = np.array([[-0.5, -np.sqrt(3)/2],[ np.sqrt(3)/2, -0.5]])
        krot = R_phys @  k
        for j, state in enumerate(self.representatives):
            #print("rep", j)
            sl = state['sl']
            #print(sl)
            if sl[0] != 0:

                phase_rot = 1/2*(np.sqrt(3)*krot[0]-3*krot[1])
            else:
                #print('state:', j, 'rotating around occupied site!')
                phase_rot = 0
            state_rot = self.rotate_state(state)
            sl_rot = state_rot['sl']
            a = self.state_2_list_entry(state_rot)
            found, i = self.basis.search(a)
            if found:
                (rep, _, m) = self.is_representative[i]
            else:
                raise ValueError("Rotation error: rotated state not in basis!")
            if not rep:
                #print("not rep, changing holes")
                la = (state_rot['hole_pos'][1] - self.depth - 1)
                if sl_rot[0] != sl_rot[1]:
                    la[0] += 2*(sl[0]-0.5)
                r1=la[0]
                r2=(la[1]-la[0])//2
                rx = np.sqrt(3)*r2+np.sqrt(3)/2*r1
                ry = -3/2*r1
                d_phys = np.array([rx, ry])
                phase_ex = np.pi + krot[0]*d_phys[0]+krot[1]*d_phys[1]
            else:
                #print("is representative")
                phase_ex = 0
            #print("full phase", (phase_rot+phase_ex)/np.pi)
            Rm[m,j]=np.exp(-1j*(phase_rot+phase_ex))
            #print("Rm[",m,j,"]=", Rm[m,j])
        self.rot_mat = Rm
    
    @staticmethod
    def mirror_y_state(state):
        lat = state['lat']
        hole_pos = state['hole_pos']

        center = (lat.shape[0]) //2

        return {'lat': lat[::-1, :], 'hole_pos': [hole_pos[0], center + np.array([-1, 1]) * (hole_pos[1]-center)], 'seq':[]}
    
    @staticmethod
    def mirror_x_state(state):
        lat = state['lat']
        hole_pos = state['hole_pos']

        center = (lat.shape[0]) //2

        return {'lat': lat[:, ::-1], 'hole_pos': [hole_pos[0], center + np.array([1, -1]) * (hole_pos[1]-center)], 'seq':[]}
