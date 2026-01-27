import os, glob, time, h5py, warnings
import numpy as np
import scipy.sparse as sp
from libra_py import units, data_stat, influence_spectrum, data_conv

# --- CCR: headless plotting ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from liblibra_core import *
from libra_py.workflows.nbra import step3
import libra_py.packages.cp2k.methods as CP2K_methods
#from IPython.display import clear_output
import multiprocessing as mp

import util.libutil as comn
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.workflows.nbra.decoherence_times as decoherence_times

# from recipes import fssh_nbra
# from recipes import dish_nbra, fssh_nbra, fssh2_nbra, gfsh_nbra, ida_nbra, mash_nbra, msdm_nbra

# =========================
# Plot 1: KS energies
# =========================
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from libra_py import units

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(3.21*2, 2.41*2))

# ===== Reading the energies
energies = []
for i in range(1000, 5000):
    energy = np.diag(sp.load_npz(f'/path/to/step2/dft_plus_u/res/E_ks_{i}.npz').toarray())
    energy_shape = int(energy.shape[0] / 2)
    energies.append(energy[0:energy_shape])

print(energy.shape)
energies = np.array(energies)
print(energies.shape)
energies = energies * units.au2ev

# Build a simple time axis (frame index). If you know dt_fs, replace with: t = np.arange(energies.shape[0]) * dt_fs
t = np.arange(energies.shape[0])

# ===== Plotting the energy vs time
for j in range(energies.shape[1]):
    if j == 100:
        color = 'green'
    elif 33 <= j <= 99:
        color = 'blue'
    elif 101 <= j < 142:
        color = 'red'
    else:
        color = 'gray'
    plt.plot(t, energies[:, j], color=color)

plt.xlabel('Time, fs')   # change to 'Time (index)' if t is just the index
plt.ylabel('Energy, eV')
plt.tight_layout()
plt.savefig("ks_energies.png", dpi=300)
plt.close()

# =========================
# Active-space limiting
# =========================
params_active_space = {
    'lowest_orbital': 248-100, 'highest_orbital': 248+50, 'num_occ_orbitals': 68, 'num_unocc_orbitals': 41,
    'path_to_npz_files': '/path/to/step2/dft_plus_u/res', 'logfile_directory': '/path/to/step2/dft_plus_u/all_logfiles',
    'path_to_save_npz_files': os.getcwd()+'/new_res_step22-5000'
}
# ensure output dir exists on CCR
os.makedirs(os.getcwd()+'/new_res_step22-5000', exist_ok=True)

new_lowest_orbital, new_highest_orbital = step3.limit_active_space(params_active_space)

# =========================
# SD-NACs (many-body, libint)
# =========================
params_mb_sd = {
          'lowest_orbital': new_lowest_orbital, 'highest_orbital': new_highest_orbital, 
          'num_occ_states': 6, 'num_unocc_states': 5,
          'isUKS': 0, 'number_of_states': 130, 'tolerance': 0.0, 'verbosity': 0, 'use_multiprocessing': True, 'nprocs': 8,
          'is_many_body': True, 'time_step': 1.0, 'es_software': 'cp2k',
          'path_to_npz_files': '/projects/academic/cyberwksp21/Students/layla/project/active_space_step3/new_res_step22-5000',
          'logfile_directory': '/projects/academic/cyberwksp21/Students/layla/project/step22/dft_plus_u/all_logfiles',
          'path_to_save_sd_Hvibs': os.getcwd()+'/res-sd-68-occ-41-unocc',
          'outdir': os.getcwd()+'/res-sd-68-occ-41-unocc', 'start_time': 1000, 'finish_time': 4998, 'sorting_type': 'energy',
         }

# CCR: make sure output dirs exist
os.makedirs(os.getcwd()+'/res-sd-68-occ-41-unocc', exist_ok=True)
os.makedirs(os.getcwd()+'/res-sd-68-occ-41-unocc', exist_ok=True)

# CCR: respect SLURM cores if available
try:
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    params_mb_sd['nprocs'] = min(params_mb_sd['nprocs'], slurm_cpus)
except Exception:
    pass

step3.run_step3_sd_nacs_libint(params_mb_sd)
# clear_output()

# =========================
# Read time-overlap diagonals
# =========================
time_overlaps = []
for i in range(1000, 4995):
    St = sp.load_npz(f'res-sd-68-occ-41-unocc/St_ci_{i}_re.npz').todense().real
    time_overlaps.append(np.diag(St))
time_overlaps = np.array(time_overlaps)

# =========================
# Plot 2: |<Ψ_GS(t)|Ψ_GS(t+Δt)>|
# =========================
plt.figure(figsize=(3.21*2,2.41*2))
plt.title('$\\|\\langle\\Psi_{GS}(t)\\mid\\Psi_{GS}(t+\\Delta t)\\rangle\\|$ - num_occ_orbitals: 68, num_unocc_orbitals: 41')
plt.plot(np.arange(time_overlaps.shape[0]), np.abs(time_overlaps[:,0]))
plt.xlabel('Time, fs')
plt.ylabel('Time-overlap absolute value')
plt.tight_layout()
plt.savefig("time_overlap_gs.png", dpi=300)
plt.close()

# =========================
# Plot 3: |det(St)|
# =========================
plt.figure(figsize=(3.21*2,2.41*2))
determinants = []
for i in range(1000, 4995):
    # unify folder name with earlier outputs: use '...unocc1'
    St = sp.load_npz(f'res-sd-68-occ-41-unocc/St_ci_{i}_re.npz').todense().real
    determinants.append(np.linalg.det(St))
plt.plot(np.arange(len(determinants)), np.abs(determinants))
plt.xlabel('Time, fs')
plt.ylabel('|Determinant|')
plt.title('68 occupied orbitals and 41 unoccupied orbitals')
plt.tight_layout()
plt.savefig("time_overlap_det.png", dpi=300)
plt.close()

