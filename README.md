
# Accelerating VQEs with Quantum Natural Gradient 

This project repo was created as part of the [Quantum Open Source Foundation (QOSF)](http://qosf.org) Mentorship Program.

Recently, several works, namely [Stokes et al.](https://arxiv.org/abs/1909.02108), have proposed and investigated the use of "quantum natural gradients" (QNG) to accelerate the optimization step of variational quantum algorithms.

To gain an in-depth and intuitive explanation of quantum natural gradients, check out the following blogposts:

1. [Rethinking Gradient Descent With Quantum Natural Gradient](https://medium.com/@ziyu.lili.maggie/rethinking-gradient-descent-with-quantum-natural-gradient-330da14f621) by Maggie Li

2. [Gradient Descent from the Ground Up](https://medium.com/@lana.bozanic/quantum-natural-gradient-from-the-ground-up-983db57cbf6) by Lana Bozanic

In addition, we provide several tutorials in this repo for running and analyzing VQE calculations of small systems using quantum natural gradient. We implemented the code in [PennyLane](https://pennylane.ai/) and used routines from [QuTiP](http://qutip.org/) for visualization.

### Tutorials

We investigate the following systems:

1. Single qubit case (where we can visualize the optimization paths using the Bloch sphere)
2. H2 molecule

and ran VQE calculations using "vanilla" gradient descent and gradient descent that uses quantum natural gradients.
We provide several methods for visualizing the performance and optimization paths, and we empirically explore the robustness of QNG to parameter initialization.
