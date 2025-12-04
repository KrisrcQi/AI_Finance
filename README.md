# AI_Finance_2025-26
This repository provides the required python files and dataset for tutorial 1 (Multi-Layer Perceptron) and tutorial 2 (Genetic Algorithm) for the course Artificial Intelligence in Finance (ACCFIN5230) at Adam Smith Business School, University of Glasgow, 2025-26.

## Python Code
The lab1 uses the MLPRegressor function from sklearn package to design MLP.\
The lab2 uses the GA function from pygad package to performe genetic algorithm.

## Dataset
The lab1 code is conducted by [loan_data](Lab1_MLP/loan.csv)\
The lab2 code uses random generating data set by uniform (x) and normal (y) distribution

## MLP Design
![image](Lab1_MLP/figures/MPL(2,).png)

## GA Design
The lab2 design a simple genetic algorithm by using actual value without encrypto. However, the common encrypto method could use binary, hash, and permutation.
 - Max iteration = 100 
 - Selection operator (parent selection): 'sss' → "Steady State Selection" 
 - Crossover type: single_point 
 - Mutation type: random
