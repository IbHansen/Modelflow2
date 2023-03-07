# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:46:09 2023
A experimental thing to find the minimal feedback vertex set
in order to improve solution speed eventualy 

@author: ibhan
"""

import networkx as nx

from modelclass import model


mken,kenbaseline = model.modelload('KENmod20230209elec2.pcim',run=1)
# Create your NetworkX graph here
G = mken.endograph.copy()
if 1: # only look at the strong component
    blocktypes = mken._superstrongtype
    blocks = mken._superstrongblock
    strongblock = blocks[1]
    
    nodes_to_remove = [n for n in G.nodes if n not in strongblock ]
    G.remove_nodes_from(nodes_to_remove)


#%%
import networkx as nx

def approx_mfvs(G):
    # Create a copy of the input graph
    G_copy = G.copy()
    
    if nx.is_directed_acyclic_graph(G_copy):
        return set() 
    
    # Initialize the removed nodes sets
    removed_nodes = set()
    prev_removed_nodes = set()

    # Iterate until the graph is empty or a stable set of removed nodes is found
    while len(G_copy) > 0:
        # Compute the approximate feedback vertex set of the remaining graph
        fvs = nx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover(G_copy)

        # Remove all nodes in the computed FVS from the graph
        G_copy.remove_nodes_from(fvs)

        # Add the removed nodes to the set of removed nodes
        removed_nodes.update(fvs)

        # If the same nodes were removed in the previous iteration, we have found a stable set
        if removed_nodes == prev_removed_nodes:
            break

        # Store the removed nodes of this iteration for the next iteration
        prev_removed_nodes = removed_nodes.copy()

    return removed_nodes

# Call the approx_mfvs function
mfvs = approx_mfvs(G)
G_test = G.copy()
G_test.remove_nodes_from(mfvs)

# Print the resulting graph and the feedback vertex set
print("Approximate minimal feedback vertex set:", mfvs)
print("Approximate minimal feedback length    :", len(mfvs))
print("Original length                        :", len(G))
print("The remaning part is a DAG             :",nx.is_directed_acyclic_graph(G_test))


#%%
import networkx as nx
import random

def approx_feedback_vertex_set(G, pop_size=50, iterations=100, elite_size=5, mutation_rate=0.1):
    """
    Returns an approximate feedback vertex set of the input graph G using a genetic algorithm.
    """
    def initial_population(node_set, pop_size):
        """
        Generates an initial population of sets of nodes to remove from the graph.
        """
        population = []
        for i in range(pop_size):
            nodes_to_remove = set(random.sample(node_set, int(len(node_set) * 0.1)))
            population.append(nodes_to_remove)
        return population
    
    def fitness(G, node_set):
        """
        Calculates the fitness of a set of nodes to remove from the graph.
        """
        H = G.copy()
        H.remove_nodes_from([node for node in H if node not in node_set])
        try:
            return len(nx.find_cycle(H))
        except nx.NetworkXNoCycle:
            return float('inf')
        
    def crossover(parent1, parent2):
        """
        Generates a child set of nodes by combining the parents' sets.
        """
        child = set()
        for node in parent1:
            if random.random() < 0.5:
                child.add(node)
        for node in parent2:
            if random.random() < 0.5 and node not in child:
                child.add(node)
        return child
    
    def mutate(node_set, mutation_rate):
        """
        Mutates a set of nodes by randomly adding or removing nodes.
        """
        mutated_set = node_set.copy()
        for node in node_set:
            if random.random() < mutation_rate:
                mutated_set.remove(node)
                if random.random() < 0.5:
                    mutated_set.add(random.choice(list(G.neighbors(node))))
        if random.random() < mutation_rate:
            mutated_set.add(random.choice(list(G.nodes())))
        return mutated_set
        
    node_set = set(G.nodes())
    best_set = node_set.copy()
    best_fitness = fitness(G, best_set)
    population = initial_population(node_set, pop_size)
    for i in range(iterations):
        # Evaluate the fitness of the population
        fitnesses = [(ind, fitness(G, ind)) for ind in population]
        fitnesses = sorted(fitnesses, key=lambda x: x[1])
        # Update the best set
        if fitnesses[0][1] < best_fitness:
            best_set = fitnesses[0][0]
            best_fitness = fitnesses[0][1]
        # Generate the next population
        new_population = [fitnesses[i][0] for i in range(elite_size)]
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(fitnesses[:pop_size], 2)
            child = crossover(parent1[0], parent2[0])
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
    return best_set

mfvs = approx_feedback_vertex_set(G)
print("Found feedback vertex set:", mfvs)