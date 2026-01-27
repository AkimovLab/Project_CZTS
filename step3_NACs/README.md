\# Step 3: Active space selection and Nonadiabatic coupling preparation



\## Purpose

Step 3 prepares the electronic structure data for NAMD. It:

\- selects a consistent \*\*active space\*\* (occupied and unoccupied orbitals) along the trajectory

\- computes and stores \*\*state energies\*\*, \*\*time overlap matrices\*\*, and \*\*nonadiabatic couplings\*\* in the basis needed for dynamics (including many body data)



The Step 3 outputs are the direct inputs for the NAMD step.



\## Inputs

From Step 2:

\- `../step2/res/`  

&nbsp; Sparse matrices for each time step (for example `E\_ks\_<t>.npz`, `S\_ks\_<t>.npz`, and related files)

\- `../step2/all\_logfiles/`  

&nbsp; CP2K log files used for excited state analysis and consistency checks  

Optional:

\- `../step2/all\_pdosfiles/`  

&nbsp; PDOS files for electronic structure interpretation









\## Outputs

\- \*\*Active space reduced data\*\* (a new `res` style folder containing reduced `npz` files for the selected orbital window)

\- \*\*Step 3 results folder\*\* containing quantities used in dynamics, such as:

&nbsp; - state energies in the chosen basis

&nbsp; - time overlap matrices

&nbsp; - nonadiabatic couplings and related matrices produced by the Step 3 workflow



These outputs are used as inputs for the NAMD step.







\## Why active space

The active space is the set of orbitals used to build the electronic state basis for couplings and dynamics. It should:

\- include orbitals relevant to the low energy excitations of interest

\- remain stable along the trajectory

\- balance accuracy and computational cost







\## After Step 3: what we can visualize (optional checks)

Using the Step 3 outputs (and Step 2 PDOS and logs), we can visualize:

\- KS orbital energies vs time (active space stability)

\- excitation energies vs time (single particle and many body views)

\- average PDOS (HOMO aligned)

\- NAC heat maps and NAC magnitude distributions

\- time overlap metrics (diagonal overlaps and determinant trends)

\- influence spectra of selected energy gaps



