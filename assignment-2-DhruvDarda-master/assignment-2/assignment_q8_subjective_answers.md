# ES654-2022 Assignment 3

Dhruv Darda - 19110012

------

> Write the answers for the subjective questions here

Dataset size: 350
Time taken for fit_autograd:  0.10791611671447754
Time taken for fit_normal:  0.0009975433349609375

Dataset size: 3500000
Time taken for fit_autograd:  0.1656637191772461
Time taken for fit_normal:  0.3717992305755615

we can see the autograd takes more time for small datasets while as we increase the size of the datset, the time is reduced for autograd.

Theoritically:

Normal Equation takes:
O(D^2 * N) + O(D^3) time, where O is order and D is the number of features while N is the number of samples.

Gradient Descent takes:
O((t + N) * D^2) time