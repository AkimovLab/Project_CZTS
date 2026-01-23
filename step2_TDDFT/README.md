\# Step 2: Electronic structure and excited state data generation (CP2K)



\## Purpose

Run CP2K calculations for geometries sampled from the Step 1 trajectory and generate the outputs needed for PDOS visualization, NTO generation, and subsequent workflow steps.



\## Inputs

\- distribute\_jobs.py: distributes the selected trajectory window into job folders (job1, job2, …) and launches Step 2.

\- run\_template.py: defines Step 2 parameters (orbital window, periodic cell, trajectory path) and runs the workflow.

\- submit\_template.slm: execution script used in each job directory.

\- es\_diag\_temp.inp: CP2K diagonalization input used for the electronic structure calculations and printed outputs.

\- es\_ot\_temp.inp: auxiliary CP2K input referenced by the workflow scripts (kept for compatibility).



\## Outputs

\- job\*/: per job run folders containing coord-\*.xyz, generated CP2K inputs (Diag\_libra-\*.inp), and CP2K output logs.

\- res/: collected numerical data across snapshots (E\_ks\_\*.npz, S\_ks\_\*.npz, St\_ks\_\*.npz).

\- all\_pdosfiles/: PDOS files collected across snapshots (Diag\_libra-<step>-czts-\*.pdos).

\- all\_logfiles/: CP2K log files collected across snapshots (step\_\*.log).



