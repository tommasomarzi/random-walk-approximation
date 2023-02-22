# Random Walk Approximation 

Code for the paper "Random Walk Approximation for Stochastic Processes on Graphs", available [here](https://www.mdpi.com/1099-4300/25/3/394)

Authors: S. Polizzi, T. Marzi, T. Matteuzzi, G. Castellani, A. Bazzani

Bibtex citation:

```
@Article{e25030394,
AUTHOR = {Polizzi, Stefano and Marzi, Tommaso and Matteuzzi, Tommaso and Castellani, Gastone and Bazzani, Armando},
TITLE = {Random Walk Approximation for Stochastic Processes on Graphs},
JOURNAL = {Entropy},
VOLUME = {25},
YEAR = {2023},
NUMBER = {3},
ARTICLE-NUMBER = {394},
URL = {https://www.mdpi.com/1099-4300/25/3/394},
ISSN = {1099-4300},
ABSTRACT = {We introduce the Random Walk Approximation (RWA), a new method to approximate the stationary solution of master equations describing stochastic processes taking place on graphs. Our approximation can be used for all processes governed by non-linear master equations without long-range interactions and with a conserved number of entities, which are typical in biological systems, such as gene regulatory or chemical reaction networks, where no exact solution exists. For linear systems, the RWA becomes the exact result obtained from the maximum entropy principle. The RWA allows having a simple analytical, even though approximated, form of the solution, which is global and easier to deal with than the standard System Size Expansion (SSE). Here, we give some theoretically sufficient conditions for the validity of the RWA and estimate the order of error calculated by the approximation with respect to the number of particles. We compare RWA with SSE for two examples, a toy model and the more realistic dual phosphorylation cycle, governed by the same underlying process. Both approximations are compared with the exact integration of the master equation, showing for the RWA good performances of the same order or better than the SSE, even in regions where sufficient conditions are not met.},
DOI = {10.3390/e25030394}
}
```

For any question or suggestion, please open an issue or send an email to the authors.

## Structure of the code

We highlight that this repository provides the code used to produce the results on the dual PdPc reported in the paper. However, since the implementation of the methods is strictly dependent on the characteristics of the model under analysis (such as number of species, rates and biological constraints), the code is not intended to be designed for an arbitrary model.

The repository is simply organized as follows:

- in the 'methods' folder are reported the scripts used to obtain the results with the methods presented in the paper, namely RWA (Random Walk Approximation), SSE (System Size Expansion), exact solution (obtained through the RK4(5) algorithm) and MUL* (standard multinomial solution obtained by linearizing the master equation at the critical point). In addition, we report also the implementation of the Gillespie algorithm. Inline documentation with references to the equations of the paper is included.
- in the 'error' folder is reported the script used for the plot of the errors included in the paper.

