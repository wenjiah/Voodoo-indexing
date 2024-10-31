# Voodoo-indexing
This is the source code for paper [A Method for Optimizing Opaque Filter Queries](https://dl.acm.org/doi/10.1145/3318464.3389766).

This uses the **MNIST** dataset as an example. In the *index_stage* folder, images are indexed by clustering. In the *query_stage* folder, the user-defined function (UDF) is trained in *MNIST_train.py*, and the Voodoo indexing algorithm is used in *compare_MNIST.py* by calling `voodoo_greedy_switch_slope_detect` function. 
