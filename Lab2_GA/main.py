# %pip install pygad

import numpy as np
import matplotlib.pyplot as plt
import pygad
from sklearn.linear_model import LinearRegression


# example 1: define a function of f1(x) = x^2 (maximization)
def f1 (x):
    return x**2

left = -1
right = 5

## plot y = x^2
xi = np.linspace(left, right, 400)
yi = f1(xi)

plt.plot(xi, yi)
plt.xlabel("x")
plt.ylabel("f1(x) = x^2")
plt.title("Function to f(x) = x^2")
plt.show()

## define the fitting function for genetic algorithm
def fitness (ga_f1, solution, solution_idx):
    x = solution[0]
    return f1(x)

## design the genetic algorithm for the example 1 of f1
ga_f1 = pygad.GA(
    num_generations = 100,
    num_parents_mating= 10,
    fitness_func = fitness,
    sol_per_pop = 20,
    num_genes = 1,
    gene_space = [{'low': left, 'high': right}],
    mutation_type = 'random'
)
ga_f1.run()
# ga_f1.summary()

solution, solution_fitness, solution_idx = ga_f1.best_solution() # locating the best result of x and y
best_x = solution[0]
best_y = solution_fitness

print("Best solution found:")
print("x =", best_x) # should close to 5
print("f(x) =", best_y) # should close to 25

## plot the fitness figure 
ga_f1.plot_fitness()

## plot the best result on the original curve
xs = np.linspace(left, right, 400)
ys = f1(xs)

plt.plot(xs, ys)
plt.scatter(solution, solution_fitness, marker='x', s=80)
plt.xlabel("x")
plt.ylabel("f(x) = x^2")
plt.title("GA result using pygad")
plt.show()


# example 2: define a minimization based on the f1
def f2 (x):
    f2 = -(f1(x))
    return f2

# plot f2(x) = -x^2
xi = np.linspace(left, right, 400)
yi = f2(xi)

plt.plot(xi, yi)
plt.xlabel("x")
plt.ylabel("f1(x) = x^2")
plt.title("Function of f(x) = -x^2")
plt.show()

# modeling the genetic algorithm for f2
def fitness_2 (ga_f2, solution, solution_idx):
    x = solution[0]
    return f2(x)

ga_f2 = pygad.GA(
    num_generations = 100,
    num_parents_mating= 10,
    fitness_func = fitness_2,
    sol_per_pop = 20,
    num_genes = 1,
    gene_space = [{'low': left, 'high': right}],
    mutation_type = 'random'
)
ga_f2.run()

solution, solution_fitness, solution_idx = ga_f2.best_solution()
best_x = solution[0]
best_y = solution_fitness

print("Best solution found:")
print("x =", best_x) # should close to 0
print("f(x) =", best_y) # should close to 0

# plot the genetic algorithm for f2
ga_f2.plot_fitness()

x2 = np.linspace(left, right, 400)
y2 = f2(x2)

plt.plot(x2, y2)
plt.scatter(solution, solution_fitness, marker='o', s = 80, color = 'red')
plt.xlabel("x")
plt.ylabel("f(x) = -x^2")
plt.title("GA result using pygad")
plt.show()


# example 3: define a function of f(x) = (x^2 + x)cos(x)
def f3 (x):
    return (x**2 + x) * np.cos(x)

left = -10
right = 10

# plot f3
x = np.linspace(left, right, 400)
y = f3(x)

plt.plot(x, y)
plt.axhline(0, linewidth=0.8)  # x-axis
plt.axvline(0, linewidth=0.8)  # y-axis
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) = (x^2 + x)cos(x)")
plt.show()

# modeling the genetic algorithm for the f3
def fitness_3 (ga_f3, solution, solution_idx):
    x = solution[0]
    return f3(x)

ga_f3 = pygad.GA(
    num_generations = 100,
    num_parents_mating= 10,
    fitness_func = fitness_3,
    sol_per_pop = 20,
    num_genes = 1,
    gene_space = [{'low': left, 'high': right}],
    mutation_type = 'random'
)
ga_f3.run()

solution, solution_fitness, solution_idx = ga_f3.best_solution()
best_x = solution[0]
best_y = solution_fitness

print("Best solution found:")
print("x =", best_x)
print("f(x) =", best_y)

# plot the fitness figure for GA_f3
ga_f3.plot_fitness()

x3 = np.linspace(left, right, 400)
y3 = f3(x3)

plt.plot(x3, y3)
plt.scatter(solution, solution_fitness, marker='o', s = 80, color = 'red')
plt.xlabel("x")
plt.ylabel("f(x) = -x^2")
plt.title("GA result using pygad")
plt.show()

# example 4: OLS regression and GA approach

## generating dataset for OLS: y = beta1 + beta2 * x: 
""" 
variables:
    x = uniform distribution
    y = normal distribution        
"""

np.random.seed(123)
y = np.random.normal(size=100)
x = np.random.uniform(size=100)

# define the Fitness function, setting beta as an array= [beta0, beta1], GA will optimizte them together
"""
Fitness function could be the objective function in the most of cases, but sometimes, it also could be something else:
  eg. RSS (coefficient determinator)
"""
def f4(beta, x, y):
    beta0, beta1 = beta
    y_hat = beta0 + beta1 * x
    rss = -np.sum((y - y_hat)**2)   # negative RSS (maximize this)
    return rss

# modeling the genetic algorithm for OLS (f4)
def fitness_4(ga_OLS, solution, solution_idx):
    return f4(solution, x, y)

left, right = -10, 10

ga_OLS = pygad.GA(
    num_generations = 100,
    num_parents_mating = 10,
    fitness_func = fitness_4,
    sol_per_pop = 20,
    num_genes = 2,   # beta0 and beta1
    gene_space = [
        {'low': left, 'high': right},  # beta0
        {'low': left, 'high': right}   # beta1
    ],
    mutation_type = 'random'
)

ga_OLS.run()

solution, solution_fitness, solution_idx = ga_OLS.best_solution()
beta0_hat, beta1_hat = solution

print("GA estimated parameters:")
print("beta0 =", beta0_hat)
print("beta1 =", beta1_hat)
print("fitness (negative RSS) =", solution_fitness)

# Compare with standard linear regression
X = x.reshape(-1, 1)
linreg = LinearRegression().fit(X, y)
print("\nOLS estimated parameters:")
print("beta0 =", linreg.intercept_) # should has a similar result as ga approached 
print("beta1 =", linreg.coef_[0]) # should has a similar result as ga approached as well.





