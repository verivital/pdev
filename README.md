# pdev
pde verification toolbox

Tested on Ubuntu 16.0.4

    Tool installation

    0) Step 0: clone tool from github: git clone https://github.com/trhoangdung/pdev.git

    1) Step 1: install scipy, numpy, sympy and matplotlib
       command: sudo pip install scipy numpy sympy matplotlib

    2) Step 2: add path to .bashrc file
       command (for example): export PYTHONPATH="${PYTHONPATH}:/home/trhoangdung/pdev"

Reproduce experimental results:

    Go to Example folder, the experimental result concludes:

    1) Discrete reachable set computation and figures : command: python plot_dreachset.py

    2) Continuous reachable set computation and figures: command: python plot_creachset.py

    3) Verify safety property and produce unsafe trace: command: python verify_safety.py

    4) Error analysis with different sizes of time step and space step: command: python plot_err_vs_spacestep.py plot_err_vs_timestep.py

    5) Computation time complexity analysis: command: python computation_time.py


All figures in the paper and .dat files containing data for the tables shoule be reproduced in example folder.
