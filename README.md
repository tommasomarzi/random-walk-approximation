# Random Walk Approximation 

Code for the paper "Random Walk Approximation for Stochastic Processes on Graphs", available at:

Authors: S. Polizzi, T. Marzi, T. Matteuzzi, G. Castellani, A. Bazzani

Bibtex citation:

```

```

For any question or suggestion, please open an issue or send an email to the authors.

## Structure of the code

We highlight that this repository provides the code used to produce the results on the dual PdPc reported in the paper. However, since the implementation of the methods is strictly dependent on the characteristics of the model under analysis (such as number of species, rates and biological constraints), the code is not intended to be designed for an arbitrary model.

The repository is simply organized as follows:

- in the 'methods' folder are reported the scripts used to obtain the results with the methods presented in the paper, namely RWA (Random Walk Approximation), SSE (System Size Expansion), exact solution (obtained through the RK4(5) algorithm) and MUL* (standard multinomial solution obtained by linearizing the master equation at the critical point). In addition, we report also the implementation of the Gillespie algorithm. Inline documentation with references to the equations of the paper is included.
- in the 'error' folder is reported the script used for the plot of the errors included in the paper.

