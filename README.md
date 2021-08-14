This is the code accompanying the paper "Towards Practical Mean Bound for Small Samples" published in the Thirty-eighth International Conference on Machine Learning (ICML 2021).

Gurobipy is free and can be installed with: 
```
python -m pip install gurobipy
```
The main file with examples is run.py and could be run with:   
```
  python3 run.py
```

The bound takes seconds to run for 1 sample using Gurobi (implemented in bound_Gurobi_solver.py). In order to run the bounds for 100,000 samples for simulation purposes, there are 2 methods:

1/ Since the bound only takes T(z) as an input, using solvers such as Gurobi it is possible to pre-compute a table that maps T(z) to the value of the bound within a reasonable amount of time. Then one could refer to the table to retrieve the bound for every sample.

2/ We provide a faster implementation (with a manual solver that vectorize loops as matrix multiplications) that does not use pre-computed table in bound_manual_solver.py
