## Overview

This repository implements the numerical framework developed in the Master's thesis to study the strong-coupling regime of the hole-doped $SU(3)$ $t\text{--}J$ model on a two-dimensional triangular lattice. The codes calculate the low-energy spectra of doped charge carriers dressed by the three-flavor antiferromagnetic (AFM) Néel background.

The simulations cover two distinct mesonic sectors and their couplings:
* **Single-Hole Sector (`SU3_1hole_triangular.py`):** Solves the dynamics of a single magnetic polaron, treated as a bound fermionic spinon-chargon ($sc$) state.
* **Two-Hole Sector (`SU3_2hole_triangular.py`):** Solves the dynamics of two doped holes forming a bound, long-lived bosonic chargon-chargon ($cc$) molecular state.
* **$sc\text{--}cc$ Overlaps (`SU3_calc_sc_cc_overlaps_tri_grid.py`):** Computes the transition matrix elements $\mathcal{M}_{\mathbf{k}}$ governing the $sc + sc \leftrightarrow cc$ recombination.

---

## Key Theoretical & Numerical Features

<p align="center">
  <img width="43%" alt="single_hole_pos0" src="https://github.com/user-attachments/assets/1ddd4b45-0071-4c55-b0c9-5ccb28c950d5" />
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img width="43%" alt="single_hole_pos1" src="https://github.com/user-attachments/assets/96b80e98-f588-4c6b-8d04-874070119876" />
</p>

* **Geometric String Basis Construction:** Under parton fractionalization, a doped hole decomposes into a mobile, charge-carrying *chargon* and a localized, flavor-carrying *spinon*. Hopping of chargons distorts the three-sublattice Néel background, generating a string of frustrated flavor bonds that acts as an attractive, linear confining potential. Because state energies scale linearly with string length, long strings are exponentially suppressed at low energies. The codes exploit this by constructing an explicit **truncated Krylov basis** applying $\hat{\mathcal{H}}\_t$ to a hole-doped Néel state up to $l\_{\text{max}}$ times.

* **Loop Degeneracy Elimination:** As chargons propagate, closed paths (such as the 7-hop Trugman loops on the triangular lattice) restore the original AFM background while translating the hole. To prevent double-counting of identical spin configurations, both scripts track unique configurations in a sorted list and employ a fast $\mathcal{O}(\log N)$ ternary search to eliminate loop degeneracies.

* **Lee-Low-Pines (LLP) Co-Moving Frame:** To utilize translational invariance across the enlarged three-site unit cells, the system Hamiltonian is mapped into the reference frame of the moving hole via a unitary Lee-Low-Pines transformation ($\hat{U}_{\text{LLP}} = e^{-i \hat{X}_f \hat{Q}_a}$). In this co-moving frame, total momentum $\mathbf{k}$ becomes a strictly conserved quantum number, transforming the Hamiltonian into a block-diagonal form in $\mathbf{k}$-space.

* **Sparse Matrix Construction & k-Space Diagonalization:**
  * For single-hole states ($sc$), hopping ($t$), next-nearest-neighbor hopping ($t'$), transverse flavor exchanges ($J_\perp$), and Ising diagonal terms ($J_z$) are evaluated to build sparse Hamiltonian matrices $\mathcal{H}_{sc}(\mathbf{k})$ for each momentum sector.
  * For two-hole states ($cc$), the scripts enforce fermionic antisymmetrization under hole exchange before constructing the two-particle sparse matrices $\mathcal{H}_{cc}(\mathbf{k})$.
  * Sparse matrix eigensolvers (`scipy.sparse.linalg.eigsh`) diagonalize each block to extract the low-energy quasiparticle dispersions across high-symmetry paths of the Magnetic Brillouin Zone (MBZ).

<p align="center">
  <img width="85%" alt="sc_cc_dispersions" src="https://github.com/user-attachments/assets/760c5b1c-ac32-433a-9b64-1078be6a0733" />
</p>

---

## Inter-Channel Overlap Calculation

To capture the unconventional pairing mechanism of the system, the code models a two-channel scattering framework. In this framework, an open continuum of independent $sc$ polarons couples to a closed molecular $cc$ channel. Near a strong-coupling Feshbach resonance, virtual transitions into these intermediate molecular states induce an effective attractive pairing potential.

* **Eigenstate Generation & Phase Fixing:** The script diagonalizes both the single-hole ($sc$) and two-hole ($cc$) Hamiltonians over a defined $\mathbf{k}$-grid. To ensure consistent interference between states, the complex phases of the resulting eigenvectors are fixed relative to the zero-string-length reference state (`find_l0_state`).

* **Transition Matrix Elements:** The script iterates through combinations of truncated $sc$ bases and computes the momentum-dependent overlaps into the $cc$ basis. This inter-channel coupling is driven by two distinct physical processes:
  * **$t'$ Overlaps:** Evaluates transitions driven by effective next-nearest-neighbor chargon hopping.
  * **$J_\perp$ Overlaps:** Evaluates transitions driven by transverse $SU(3)$ flavor exchanges (spin-flips) along the frustrated bonds of the geometric strings.

<p align="center">
  <img width="779" height="259" alt="t&#39;_coupling" src="https://github.com/user-attachments/assets/8096405c-32d4-4198-a150-686955fcd167" />
  <br>
  <img width="779" height="259" alt="J_coupling" src="https://github.com/user-attachments/assets/855dba7d-3761-4b43-af8f-5fe840136bcf" />
</p>

* **String Exclusion Rules:** To maintain topological consistency and computational efficiency, the algorithm applies strict exclusion rules. It discards configurations where the two $sc$ strings intersect (to avoid time-ordering ambiguities) or where the combined state exceeds the maximum overlap string length (`l_max_sc_overlaps`).
* **Symmetry & Visualization:** The script utilizes KDTree-based mapping to handle $C_3$ rotational symmetries and outputs the magnitude and complex phase of the coupling channels, plotting them directly onto the Brillouin Zone using Matplotlib's `tripcolor`.

<p align="center">
  <img width="90%" alt="Overlaps_jperp_gauge_fixed" src="https://github.com/user-attachments/assets/ff9e9f3a-60a2-4d4e-91b3-eecc36f777ce" />
</p>
