import numpy as np
from genetics import Genome, Solution, Room
from helper import get_cost_of_solution
from metrics import GenerationMetrics, SingleRunMetrics, RunLog, Parameters, get_fitness_of_solution, get_cost_of_solution


class GeneticAlgorithm():
    """A genetic algorithm for the rectangle packing problem.

    Attributes:
        population_size: The size of the population.
        mutation_rate: The probability of a mutation.
        num_generations: The number of generations to run.
        crossover_rate: The probability of a crossover.
        val_size: The number of bits used to represent a value.
    """
    def __init__(self,
                 population_size: int,
                 mutation_rate: float,
                 num_generations: int,
                 crossover_rate: float,
                 val_size: int):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.val_size = val_size

    def initialize_population(self) -> list[Genome]:
        """Initialize the population randomly within constraints."""
        pop = [Genome(val_size=self.val_size,
                       mutation_rate=self.mutation_rate,
                       crossover_rate=self.crossover_rate,
                       ) for _ in range(self.population_size)]
        for indiv in pop:
            indiv.generate_genome()
        return pop
    
    def evaluate_cost(self, population: list[Genome]) -> list[float]:
        """Evaluate the cost of each solution in the population."""
        return [
            get_cost_of_solution(indiv.decode())
            for indiv in population]

    def evaluate_fitness(self, costs: list[float]) -> list[float]:
        """Evaluate the fitness of each cost in the population."""
        return [
            get_fitness_of_solution(cost)
            for cost in costs]

    def select_population(self, population: list[Genome], fitness_scores: list[float]) -> list[Genome]:
        """Select solutions for reproduction.
        Use roulette wheel selection."""
        # Normalize fitness scores
        fitness_scores = np.array(fitness_scores)
        fitness_scores /= fitness_scores.sum()

        # Select solutions for reproduction
        selected_population = []
        for _ in range(self.population_size):
            selected_population.append(
                population[np.random.choice(len(population), p=fitness_scores)]
            )
        return selected_population

    def create_new_population(self, selected_population: list[Genome]) -> list[Genome]:
        """Apply crossover and mutation to create a new population."""
        # Apply crossover
        new_population: list[Genome] = []
        for i in range(0, len(selected_population), 2):
            if Genome.crossover_rate < np.random.random():
                # Don't apply crossover
                new_population.append(selected_population[i])
                new_population.append(selected_population[i + 1])
            else:
                # Apply crossover
                new_population.extend(
                    selected_population[i].crossover(selected_population[i + 1])
                )
        # Apply mutation
        for indiv in new_population:
            indiv.mutate()
        return new_population

    def find_best_solution(self, population: list[Genome]) -> Solution:
        """Find and return the best solution."""
        costs = self.evaluate_cost(population)
        fitness_scores = self.evaluate_fitness(costs)
        best_index = np.argmax(fitness_scores)
        return population[best_index].decode()

    def run(self) -> tuple[Genome, SingleRunMetrics]:
        """Run the genetic algorithm."""
        # Initialize the population randomly within constraints
        population = self.initialize_population()
        # Initialize the run log
        run_log = SingleRunMetrics()

        # Main genetic algorithm loop
        for generation in range(self.num_generations):
            # Evaluate the cost of each solution in the population
            cost_scores = self.evaluate_cost(population)

            # Evaluate the fitness of each solution in the population
            fitness_scores = self.evaluate_fitness(cost_scores)

            # Select solutions for reproduction
            selected_population = self.select_population(population, fitness_scores)

            # Apply crossover and mutation to create a new population
            new_population = self.create_new_population(selected_population)

            # Replace the old population with the new one
            population = new_population

            best_cost = min(cost_scores)
            best_fitness = max(fitness_scores)
            avg_cost = np.mean(cost_scores)
            avg_fitness = np.mean(fitness_scores)
            worst_cost = max(cost_scores)
            worst_fitness = min(fitness_scores)

            # Log the metrics for this generation
            run_log.add_generation_metrics(
                GenerationMetrics(
                    generation=generation,
                    best_cost=best_cost,
                    avg_cost=avg_cost,
                    worst_cost=worst_cost,
                    best_fitness=best_fitness,
                    avg_fitness=avg_fitness,
                    worst_fitness=worst_fitness,
                    num_evals=len(cost_scores)
                )
            )
        # End of main genetic algorithm loop
        print("Done running genetic algorithm.")
        # return the best solution
        best_solution = self.find_best_solution(population)

        # Account for the finding of the best solution
        run_log.final_evals = len(population)

        # Return the best solution and the run log
        return best_solution, run_log
