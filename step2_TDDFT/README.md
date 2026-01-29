# Step 2: Electronic structure and excited state data generation (CP2K)



## Purpose

Run CP2K calculations for geometries sampled from the Step 1 trajectory and generate the outputs needed for PDOS visualization, NTO generation, and subsequent workflow steps.



## Inputs

- `distribute_jobs.py`: distributes the selected trajectory window into job folders (job1, job2, …) and launches Step 2.

- `run_template.py`: defines Step 2 parameters (orbital window, periodic cell, trajectory path) and runs the workflow.

- `submit_template.slm`: execution script used in each job directory.

- `es_diag_temp.inp`: CP2K diagonalization input used for the electronic structure calculations and printed outputs.

- `es_ot_temp.inp`: auxiliary CP2K input referenced by the workflow scripts (kept for compatibility).




## Outputs

- `job*`: per job run folders containing `coord-*.xyz`, generated CP2K inputs (`Diag_libra-*.inp`), and CP2K output logs.

- `res`: collected numerical data across snapshots (`E_ks_*.npz`, `S_ks_*.npz`, `St_ks_*.npz`).

- `all_pdosfiles`: PDOS files collected across snapshots (`Diag_libra-<step>-czts-*.pdos`).

- `all_logfiles`: CP2K log files collected across snapshots (`step_*.log`).




