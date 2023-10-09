from __future__ import annotations
from dataclasses import dataclass
from genetics import Solution
import numpy as np


# Formula (for areas):
# cost = Living + Kitchen * 2 + Bath * 2 + Hall + Bedroom1 + Bedroom2 + Bedroom3
min_cost = 632.5
max_cost = 1245.5


def get_cost_of_solution(solution: Solution) -> float:
    """Calculates the cost of a solution.

    Args:
        solution: The solution to calculate the cost of.

    Returns:
        Number: The cost of the solution.
    """
    cost = 0
    for room in solution.rooms:
        width = solution.rooms[room].width
        height = solution.rooms[room].height
        if room == "Kitchen" or room == "Bath":
            cost += width * height * 2
        else:
            cost += width * height
    assert min_cost <= cost <= max_cost
    return cost


def get_fitness_of_solution(cost: float) -> float:
    """Calculates the fitness of a solution.

    Args:
        solution: The solution to calculate the fitness of.

    Returns:
        Number: The fitness of the solution.
    """
    return max_cost - cost

@dataclass
class GenerationMetrics():
    generation: int
    best_cost: float
    avg_cost: float
    worst_cost: float
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    num_evals: int

    def __repr__(self):
        return f"Generation: {self.generation}, \
                Best Cost: {self.best_cost}, \
                Avg Cost: {self.avg_cost}, \
                Worst Cost: {self.worst_cost}, \
                Best Fitness: {self.best_fitness}, \
                Avg Fitness: {self.avg_fitness}, \
                Worst Fitness: {self.worst_fitness}, \
                Num Evals: {self.num_evals}"


class SingleRunMetrics():
    def __init__(self):
        self.generation_metrics = []
        self.final_evals = 0

    def add_generation_metrics(self, generation_metrics: GenerationMetrics) -> SingleRunMetrics:
        self.generation_metrics.append(generation_metrics)
        return self

    def get_best_cost(self) -> float:
        return self.generation_metrics[-1].best_cost

    def get_best_fitness(self) -> float:
        return self.generation_metrics[-1].best_fitness

    def get_avg_cost(self) -> float:
        return self.generation_metrics[-1].avg_cost

    def get_avg_fitness(self) -> float:
        return self.generation_metrics[-1].avg_fitness

    def get_worst_cost(self) -> float:
        return self.generation_metrics[-1].worst_cost

    def get_worst_fitness(self) -> float:
        return self.generation_metrics[-1].worst_fitness

    def get_num_evals(self) -> int:
        return self.final_evals + sum([gen.num_evals for gen in self.generation_metrics])


@dataclass
class Parameters():
    population_size: int
    mutation_rate: float
    num_generations: int
    crossover_rate: float
    val_size: int


class RunLog():
    def __init__(self):
        self.runs: list[SingleRunMetrics] = []
        self.parameters: Parameters = None
        self.cached_quality: float = None
        self.cached_reliability: float = None
        self.cached_speed: float = None

    def set_parameters(self, parameters: Parameters) -> RunLog:
        self.parameters = parameters
        return self

    def add_run(self, run: SingleRunMetrics) -> RunLog:
        self.runs.append(run)
        self.cached_quality = None
        self.cached_reliability = None
        self.cached_speed = None
        return self

    # Quality = percentage distance from optimum (Average of best over all runs)
    def calculate_quality(self) -> float:
        if self.cached_quality is None:
            mean = np.mean([run.get_best_cost() for run in self.runs])
            range = max_cost - min_cost
            self.cached_quality = (mean - min_cost) / range
        return self.cached_quality

    # Reliability = Percentage of runs you get within Quality -- out of total number of runs.
    def calculate_reliability(self) -> float:
        if self.cached_reliability is None:
            num_within_quality = 0
            for run in self.runs:
                if run.get_best_cost() <= self.calculate_quality():
                    num_within_quality += 1
            self.cached_reliability = num_within_quality / len(self.runs)
        return self.cached_reliability

    # Speed = Average number of evaluations to get within Quality
    def calculate_speed(self) -> float:
        if self.cached_speed is None:
            total_num_evals = 0
            num_within_quality = 0
            for run in self.runs:
                if run.get_best_cost() <= self.calculate_quality():
                    num_within_quality += 1
                    total_num_evals += run.get_num_evals()
            if num_within_quality == 0:
                self.cached_speed = 0
            else:
                self.cached_speed = total_num_evals / num_within_quality
        return self.cached_speed

    def __repr__(self):
        return f"Quality: {self.calculate_quality()}, \
                Reliability: {self.calculate_reliability()}, \
                Speed: {self.calculate_speed()}"
