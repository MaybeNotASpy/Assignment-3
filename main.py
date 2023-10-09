from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
from metrics import RunLog, Parameters
from GA import GeneticAlgorithm
import os
from multiprocessing import Pool
import pickle

SEARCH = False

def run_param_set(param_set: int):
    params = Parameters(
        np.random.randint(5, 100) * 2,
        np.random.uniform(0, 1),
        np.random.randint(10, 200),
        np.random.uniform(0, 1),
        np.random.randint(5, 16))
    # Make directory for this param set.
    if not os.path.exists(f"results/param_set{param_set}"):
        os.mkdir(f"results/param_set{param_set}")
    run_log = RunLog().set_parameters(params)
    dir = f"results/param_set{param_set}"
    for i in range(10):    
        print(f"Running param set {param_set}, run {i}")
        ga = GeneticAlgorithm(params.population_size,
                                params.mutation_rate,
                                params.num_generations,
                                params.crossover_rate,
                                params.val_size)
        best_solution, run = ga.run()
        run_log.add_run(run)
        print(f"Finished param set {param_set}, run {i}")
    # End of run loop.

    # Save the run log
    try:
        with open(f"{dir}/run_log.pkl", "w") as f:
            for i in range(len(run_log.runs)):
                f.write("Run {i}:\n}")
                for j in range(len(run_log.runs[i].generation_metrics)):
                    f.write(f"Generation {j}:\n")
                    f.write(f"Best Cost: {run_log.runs[i].generation_metrics[j].best_cost}\n")
                    f.write(f"Avg Cost: {run_log.runs[i].generation_metrics[j].avg_cost}\n")
                    f.write(f"Worst Cost: {run_log.runs[i].generation_metrics[j].worst_cost}\n")
                    f.write(f"Best Fitness: {run_log.runs[i].generation_metrics[j].best_fitness}\n")
                    f.write(f"Avg Fitness: {run_log.runs[i].generation_metrics[j].avg_fitness}\n")
                    f.write(f"Worst Fitness: {run_log.runs[i].generation_metrics[j].worst_fitness}\n")
                    f.write(f"Num Evals: {run_log.runs[i].generation_metrics[j].num_evals}\n")
                f.write(f"Final Evals: {run_log.runs[i].final_evals}\n")
                f.write(f"Best Cost: {run_log.runs[i].get_best_cost()}\n")
                f.write(f"Avg Cost: {run_log.runs[i].get_avg_cost()}\n")
                f.write(f"Worst Cost: {run_log.runs[i].get_worst_cost()}\n")
                f.write(f"Best Fitness: {run_log.runs[i].get_best_fitness()}\n")
                f.write(f"Avg Fitness: {run_log.runs[i].get_avg_fitness()}\n")
                f.write(f"Worst Fitness: {run_log.runs[i].get_worst_fitness()}\n")
                f.write(f"Num Evals: {run_log.runs[i].get_num_evals()}\n")
            f.write(f"Speed: {run_log.calculate_speed()}\n")
            f.write(f"Quality: {run_log.calculate_quality()}\n")
            f.write(f"Reliability: {run_log.calculate_reliability()}\n")
            f.write(f"Population Size: {params.population_size}\n")
            f.write(f"Mutation Rate: {params.mutation_rate}\n")
            f.write(f"Number of Generations: {params.num_generations}\n")
            f.write(f"Crossover Rate: {params.crossover_rate}\n")           
    except:
        print("Failed to save run log.")


    # Save the best solution.
    try:
        with open(f"{dir}/best_solution.pkl", "w") as f:
            f.write(f"{best_solution}\n")
    except:
        print("Failed to save best solution.")
        
    
    # Plot the results.
    # Want to plot:
    # Plot1: Best cost over generations
    # Plot1: Avg cost over generations
    # Plot1: Worst cost over generations
    # Plot2: Best fitness over generations
    # Plot2 Avg fitness over generations
    # Plot2 Worst fitness over generations
    # Plot3: Number of evaluations over generations

    # Plot 1
    best_costs = []
    avg_costs = []
    worst_costs = []
    for gen in run_log.runs[-1].generation_metrics:
        best_costs.append(gen.best_cost)
        avg_costs.append(gen.avg_cost)
        worst_costs.append(gen.worst_cost)
    plt.plot(best_costs, label="Best")
    plt.plot(avg_costs, label="Avg")
    plt.plot(worst_costs, label="Worst")
    plt.legend()
    plt.title("Costs")
    plt.savefig(f"{dir}/costs.png")
    plt.clf()

    # Plot 2
    best_fitnesses = []
    avg_fitnesses = []
    worst_fitnesses = []
    for gen in run_log.runs[-1].generation_metrics:
        best_fitnesses.append(gen.best_fitness)
        avg_fitnesses.append(gen.avg_fitness)
        worst_fitnesses.append(gen.worst_fitness)
    plt.plot(best_fitnesses, label="Best")
    plt.plot(avg_fitnesses, label="Avg")
    plt.plot(worst_fitnesses, label="Worst")
    plt.legend()
    plt.title("Fitnesses")
    plt.savefig(f"{dir}/fitnesses.png")
    plt.clf()

    # Plot 3
    num_evals = []
    cum_evals = 0
    for gen in run_log.runs[-1].generation_metrics:
        cum_evals += gen.num_evals
        num_evals.append(cum_evals)
    plt.plot(num_evals)
    plt.title("Number of Evaluations")
    plt.savefig(f"{dir}/evals.png")
    plt.clf()

    # Save the speed, quality, and reliability.
    with open(f"{dir}/metrics{i}.txt", "w") as f:
        f.write(f"Speed: {run_log.calculate_speed()}\n")
        f.write(f"Quality: {run_log.calculate_quality()}\n")
        f.write(f"Reliability: {run_log.calculate_reliability()}\n")

    # Save the parameters.
    with open(f"{dir}/params{i}.txt", "w") as f:
        f.write(f"Population Size: {params.population_size}\n")
        f.write(f"Mutation Rate: {params.mutation_rate}\n")
        f.write(f"Number of Generations: {params.num_generations}\n")
        f.write(f"Crossover Rate: {params.crossover_rate}\n")
        f.write(f"Val Size: {params.val_size}\n")
# End of run_param_set.

def final_run():
    dir = "results/final_run"
    if not os.path.exists(dir):
        os.mkdir(dir)

    params = Parameters(
        180,
        0.6140611174108054,
        175,
        0.7197212695511637,
        12)
    
    ga = GeneticAlgorithm(params.population_size,
                            params.mutation_rate,
                            params.num_generations,
                            params.crossover_rate,
                            params.val_size)
    best_solution, run = ga.run()
    
    # Save the run log
    with open(dir + "/run.log", "w") as f:
        f.write(f"Best Cost: {run.get_best_cost()}\n")
        f.write(f"Avg Cost: {run.get_avg_cost()}\n")
        f.write(f"Worst Cost: {run.get_worst_cost()}\n")
        f.write(f"Best Fitness: {run.get_best_fitness()}\n")
        f.write(f"Avg Fitness: {run.get_avg_fitness()}\n")
        f.write(f"Worst Fitness: {run.get_worst_fitness()}\n")
        f.write(f"Num Evals: {run.get_num_evals()}\n")

    # Save the best solution.
    with open(dir + "/best_solution.pkl", "w") as f:
        f.write(f"{best_solution}\n")

    # Plot the results.
    # Want to plot:
    # Plot1: Best cost over generations
    # Plot1: Avg cost over generations
    # Plot1: Worst cost over generations
    # Plot2: Best fitness over generations
    # Plot2 Avg fitness over generations
    # Plot2 Worst fitness over generations
    # Plot3: Number of evaluations over generations
    
    # Plot 1
    best_costs = []
    avg_costs = []
    worst_costs = []
    for gen in run.generation_metrics:
        best_costs.append(gen.best_cost)
        avg_costs.append(gen.avg_cost)
        worst_costs.append(gen.worst_cost)
    plt.plot(best_costs, label="Best")
    plt.plot(avg_costs, label="Avg")
    plt.plot(worst_costs, label="Worst")
    plt.legend()
    plt.title("Costs")
    plt.savefig(dir + "/costs.png")
    plt.clf()

    # Plot 2
    best_fitnesses = []
    avg_fitnesses = []
    worst_fitnesses = []
    for gen in run.generation_metrics:
        best_fitnesses.append(gen.best_fitness)
        avg_fitnesses.append(gen.avg_fitness)
        worst_fitnesses.append(gen.worst_fitness)
    plt.plot(best_fitnesses, label="Best")
    plt.plot(avg_fitnesses, label="Avg")
    plt.plot(worst_fitnesses, label="Worst")
    plt.legend()
    plt.title("Fitnesses")
    plt.savefig(dir + "/fitnesses.png")
    plt.clf()

    # Plot 3
    num_evals = []
    cum_evals = 0
    for gen in run.generation_metrics:
        cum_evals += gen.num_evals
        num_evals.append(cum_evals)
    plt.plot(num_evals)
    plt.title("Number of Evaluations")
    plt.savefig(dir + "/evals.png")
    plt.clf()

    # Save the number of evaluations.
    with open(dir + "/metrics.txt", "w") as f:
        f.write(f"Speed: {run.get_num_evals()}\n")

def main():
    # Make results directory if it doesn't exist.
    if not os.path.exists("results"):
        os.mkdir("results")
    if SEARCH:
        # Run the GA 10 times for each set of parameters.
        with Pool(10) as p:
            p.map(run_param_set, range(10))
        # End of param set loop.
    else:
        final_run()
# End of main.


if __name__ == '__main__':
    main()
