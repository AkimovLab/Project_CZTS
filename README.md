### Folders

- `step1` – Ground-state molecular dynamics of CZTS using CP2K  
  > Used to generate thermally equilibrated geometries for excited-state calculations.

- `step2` – TDDFT and time-overlap calculations  
  > Calculates excitation energies and molecular orbitals across selected MD frames.

- `step3` – Nonadiabatic coupling calculations  
  > Computes NACs between excited states to prepare inputs for NA-MD simulations.

- `step4` – Surface hopping simulations  
  > Performs nonadiabatic molecular dynamics using surface hopping to track population evolution and relaxation dynamics.

### Note

- The steps should be completed in order: `step1`, `step2`, `step3`, and `step4`.
# Project_CZTS
