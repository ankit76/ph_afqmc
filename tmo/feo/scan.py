import numpy as np
import sys
import os

def write_afqmc_input(dt = 0.005, nsteps = 50, ndets = 100, fname = 'afqmc.json'):
  afqmc_input =  '''
{
  "system":
  {
    "integrals": "FCIDUMP_chol",
    "numAct": 18
  },
  "wavefunction":
  {
    "right": "uhf",
    "left": "multislater",
    "determinants": "dets_0.bin",
    "ndets": %i
  },
  "sampling":
  {
    "seed": %i,
    "phaseless": true,
    "dt": %f,
    "nsteps": %i,
    "nwalk": 50,
    "choleskyThreshold": 2.0e-3,
    "orthoSteps": 20,
    "stochasticIter": 500
  }
}
  '''%(ndets, np.random.randint(1, 1e6), dt, nsteps)
  f = open (fname, "w")
  f.write(afqmc_input)
  f.close()
  return


def write_dice_input(eps = 1e-5, fname = 'dice.dat'):
  dice_input =  '''nocc 22
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 22 24
end
orbitals ./FCIDUMP_can
nroots 1

#variational
schedule
0	%f
end
davidsonTol 5e-05
dE 1e-05
maxiter 10
writebestdeterminants 1000

#pt
nPTiter 0
epsilon2 1e-07
#epsilon2Large 2e-5
targetError 1e-5
sampleN 200

#misc
noio
#prefix /scratch/summit/anma2640/fep
#DoRDM
#DoSpinRDM
  '''%(eps)
  f = open(fname, "w")
  f.write(dice_input)
  f.close()
  return

os.system("export OMP_NUM_THREADS=1; rm samples.dat -f")

for ndets in [ 1, 10, 100, 1000, 10000 ]:
  print(f'ndets: {ndets}')
  write_afqmc_input(ndets=ndets)
  command = f'''
                mpirun -np 56 /projects/anma2640/VMC/dqmc/VMC/bin/DQMC afqmc.json > afqmc_{ndets}.out;
                python /projects/anma2640/VMC/dqmc/VMC/scripts/blocking.py samples.dat 50 > blocking_{ndets}.out;
                mv samples.dat samples_{ndets}.dat
             '''
  os.system(command)

os.system("export OMP_NUM_THREADS=36")
