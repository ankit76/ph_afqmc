import numpy as np
import sys
import os

def write_afqmc_input(dt = 0.005, nsteps = 50, ndets = 100, fname = 'afqmc.json'):
  afqmc_input =  '''
{
  "system":
  {
    "integrals": "FCIDUMP_chol",
    "numAct": 50
  },
  "wavefunction":
  {
    "right": "rhf",
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
    "nwalk": 25,
    "choleskyThreshold": 2.0e-3,
    "orthoSteps": 20,
    "stochasticIter": 200
  }
}
  '''%(ndets, np.random.randint(1, 1e6), dt, nsteps)
  f = open (fname, "w")
  f.write(afqmc_input)
  f.close()
  return


def write_dice_input(eps = 1e-5, ndets = 1000, fname = 'dice.dat'):
  dice_input ='''nocc 14
0 1 2 3 4 5 6 7 8 9 10 11 12 13
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
#writebestdeterminants
ndetsRestart %i

#pt
nPTiter 0
epsilon2 1e-07
#epsilon2Large 2e-5
targetError 1e-5
sampleN 200

#misc
restart
noio
DoOneRDM
#prefix /scratch/summit/anma2640/nh3
#DoRDM
#DoSpinRDM
'''%(eps, ndets)
  f = open(fname, "w")
  f.write(dice_input)
  f.close()
  return

nproc = 56
os.system("export OMP_NUM_THREADS=1; rm samples.dat -f")

#for ndets in [ 50000 ]:
for ndets in [ 10000, 1000, 100, 10, 1 ]:
  print(f'ndets: {ndets}', fname=f'dice_rdm_{ndets}.dat')
  # dice
  write_dice_input(ndets=ndets)
  command = f"mpirun -np 1 /projects/anma2640/newDice/Dice/Dice dice_rdm_{ndets}.dat > dice_rdm_{ndets}.out;"
  os.system(command)
  # afqmc
  write_afqmc_input(ndets=ndets)
  command = f'''
                mpirun -np {nproc} /projects/anma2640/VMC/dqmc/VMC/bin/DQMC afqmc.json > afqmc_{ndets}.out;
                python /projects/anma2640/VMC/dqmc/VMC/scripts/blocking.py samples.dat 50 > blocking_{ndets}.out;
                mv samples.dat samples_{ndets}.dat;
                python dipole.py dipoleInts.h5 > dipole_{ndets}.out;
             '''
  os.system(command)

#os.system("export OMP_NUM_THREADS=36")
