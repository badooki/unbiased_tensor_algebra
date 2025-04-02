# Derive an unbiased estimator of any linear tensor algebra
Please read [this note](disjoint_sum.pdf) for a detailed explanation of the mathematical framework of this method.

## Problem setup
Many unbiased statistical quantities based on matrices or tensors can be written in the following form:

$$ V_{n}\coloneqq\frac{1}{\text{number of summands}}\sum_{i_{1}\neq i_{2}\neq...\neq i_{n}}T_{i_{1},i_{2},\ldots,i_{n}}. $$

This is called U-statistics, and quantities of this form needed to be derived manually by hand, since the overarching mechanism of the derivation has not been clearly defined.
In [this note](disjoint_sum.pdf), I show how U-statistics can be derived in a mechanistic manner, based on the results from combinatorics and graph theory.
This allows us to easily implement the derivation on a computer.

## Usage
Required packages:
- numpy
- opt_einsum
- JAX

The example demonstration is shown in example.ipynb.
