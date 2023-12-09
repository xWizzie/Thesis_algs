from General_functions import *
from statsmodels.stats.proportion import proportion_confint
import numpy as np
from scipy.spatial import distance
import math


def Nearest_Neighbor_Radius(learning_set, new_instance, kappa):
    distances = []
    for instance in learning_set:
        dist = distance.euclidean(instance.coords, new_instance.coords)
        distances.append(dist)
    distances.sort()
    return distances[kappa-1]


def Observations_Within_Radius(combined_set, new_instance, radius):
    neighbors = []
    for instance in combined_set:
        dist = distance.euclidean(instance.coords, new_instance.coords)
        if dist <= radius:
            neighbors.append(instance)
    return neighbors


def agree(instance_from_set, new_instance_prediction):
    if instance_from_set == new_instance_prediction:
        return 1
    else:
        return 0


def predict(N):
    label_counts = {}
    for neighbor in N:
        if neighbor.label in label_counts:
            label_counts[neighbor.label] += 1
        else:
            label_counts[neighbor.label] = 1
    predicted_label = max(label_counts, key=label_counts.get)
    return predicted_label


unacceptable_set = []
learning_set = generate_synthetic_data(100)
beta = 0.8
kappa = math.ceil(beta * len(learning_set))
SR = {}


def PECS(learning_set, new_instance, pmin, pmax):
    """
    PECS Algorithm Parameters
    ----------
    learning_set : list of DataInstance objects
        The learning set
    new_instance : DataInstance object
        The new instance to be classified
    pmin : float    
        The minimum acceptable probability
    pmax : float
        The maximum acceptable probability
    SRi : integer
        Shift Register of instance i
    x : integer 
        Number of instances in N that agree with instance i
    n : integer
        Number of attempts
    lb:  float
        Lower bound of the confidence 
    up:  float
        Upper bound of the confidence    
    """
    # r = Nearest Neighbor Radius(L,e,k)
    # N = Observations Within Radius(L u U,e,r)

    # Wouldn't this be the same as nearest_neighbors?
    r = Nearest_Neighbor_Radius(learning_set, new_instance, kappa)
    combined_set = learning_set + unacceptable_set
    N = Observations_Within_Radius(combined_set, new_instance, r)
    learning_set.append(new_instance)

    for instance in N:
        outcome = agree(instance.label, predict(N))
        if instance in SR:

            if outcome == 1:
                SR[instance]['x'] += 1
            SR[instance]['n'] += 1
        else:
            if outcome == 1:
                x = 1
            else:
                x = 0
            SR[instance] = {'x': x, 'n': 1}

        # confit = proportion_confint(x, n, alpha=0.05, method='normal')
    print(SR)

PECS(learning_set, DataInstance([5, 5], 'A'), 0.5, 0.75)
