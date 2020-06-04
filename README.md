# Working with Quantum Natural Gradient 

### Intro to the Repo
Welcome to the repo feauturing all-things QNG x VQE! All contributions to this repo were made during the Quantum Open Source Foundation (QOSF) Mentorship Program (you can check them out at: http://qosf.org !) 

The purpose of this repo was to experiment with the new quantum natural gradient descent opimization method for VQEs, as proposed in <a href = 'https://arxiv.org/abs/1909.02108'> Stokes et al. (2019)</a>, and compare its perfomance with the vanilla (original) gradient descent method.

Our findings were that in every case the Quantum Natural Gradient (QNG) method would always out-best the vanilla gradient descent method. To understand why, check out these two articles written by QOSF mentees Maggie and Lana, for an in-depth and intuitive explanation of the algorithm:

1. Maggie: Rethinking Gradient Descent With Quantum Natural Gradient
https://medium.com/@ziyu.lili.maggie/rethinking-gradient-descent-with-quantum-natural-gradient-330da14f621

2. Lana: Gradient Descent from the Group Up
https://medium.com/@lana.bozanic/quantum-natural-gradient-from-the-ground-up-983db57cbf6

or, check out the QNG tutorials in the repo!

### Tutorials in the Repo 

All the tutorials apply QNG to some Hamiltonian for the variational quantum eigensolver algorithm (VQE):

1. Single qubit case
2. H2 molecule (finding the ground state energy)
3. LiH molecule 

We have also created tutorials specific for the visualization of the optimization process for the single qubit and H2 molecules:

1. Distribution of k-runs for H2 molecule to compare QNG with diagonal approximation, QNG with block-diagonal approximation and Vanilla gradient descent 
2. Optimization 'pathway' for the single qubit visualized on a bloch sphere + visualized on the parameter space
