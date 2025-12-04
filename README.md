# AI_Finance_2025-26
This repository provides the required python files and dataset for tutorial 1 (Multi-Layer Perceptron) and tutorial 2 (Genetic Algorithm) for the course Artificial Intelligence in Finance (ACCFIN5230) at Adam Smith Business School, University of Glasgow, 2025-26.

## Python Code
Lab1 uses the MLPRegressor function from sklearn package to design MLP.\
Lab2 uses the GA function from pygad package to performe genetic algorithm.

## Dataset
The lab1 code is conducted by [loan_data](Lab1_MLP/loan.csv)\
The lab2 code uses random generating data set by uniform (x) and normal (y) distribution

## MLP Design
Lab1 designed a (2,) structured multi-layer perceptron to make prediction 
 - Dependent variable: (fico) 
 - Independent variables: (int_rate, installment, log_annual, dti)

Where the explanation of each variable is addressed in the slide.

The MLP structure is shown below:

![image](Lab1_MLP/figures/MPL(2,).png)

## GA Design
Lab2 designs a simple genetic algorithm by using actual value without encrypting/encoding.\
However, the common encrypto method could use binary, hash, and permutation.

Optimization problems:
1. max f(x) = x^2  s.t. -1 <= x <= 5
2. min f(x) = x^2  s.t. -1 <= x <= 5
3. max f(x) = (x^2+x)cos(x)  s.t. -10 <= x <= 10
4. optimizing parameter for OLS: min RSS  s.t. -10 <= beta <= 10

GAs are generally designed:
 - Max iteration = 100
 - Number of parents mating = 20
 - fitness function = objective function (some cases are not using ojective function)
 - Number of gene/parameters = 1 (for task 1-3) / 2 (for task 4)
 - Selection operator (parent selection): 'sss' → "Steady State Selection" 
 - Crossover type: single_point 
 - Mutation type: random
