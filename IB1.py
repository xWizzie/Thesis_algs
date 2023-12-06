import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math
from General_functions import *

training_data = generate_synthetic_data()

def calc_sim(x, y):
    return -math.sqrt(np.sum((np.array(x.coords) - np.array(y.coords))**2))

def IB1():
    # Generate synthetic data
    training_data = generate_synthetic_data()
    
    cd = list()
    sim_list = {}

    # Initialize classification
    classification = None
    # Iterate over each instance x in the training data
    for x in training_data:
        # 1.
        # Initialize the similarity list for this instance
        sim_list = {}

        # Iterate over each instance y in the concept description
        for y in cd:
            # Calculate the similarity between x and y
            sim_list[y] = calc_sim(x, y)
        # 2.
        # If the similarity list is not empty, classify x
        if sim_list:
            # Find the instance y in the concept description with the maximum similarity to x
            ymax = max(sim_list, key=sim_list.get)

            # 3.
            # Classify x based on the class of ymax
            if x.label == ymax.label:
                classification = 'correct'
            else:
                classification = 'incorrect'

            # Print the classification of x
            print(x.coords, classification)

        # Add x to the concept description
        cd.append(x)

    # Visualize the data
    # visualize_data(training_data, cd)