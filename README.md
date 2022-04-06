# On-the-Convergence-of-the-MCES-Algorithm-for-RL

The code for generating the numerical results can be found in the `src/` folder. 

`blackjack_mces.py` runs MCES on BlackJack. 

`cliff_walking_mces.py` and `cliff_walking_qlearning.py` have code on running MCES and Q learning in cliff_walking. `cliff_walking_mces_runner.py` and `cliff_walking_qlearning_runner.py` are used to run cliffwalking experiments. `cliff_walking_mces.sh` and `cliff_walking_qlearning.sh` are used to run experiments on the HPC (we used NYU's HPC which has SLURM system). 

