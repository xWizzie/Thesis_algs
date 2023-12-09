from General_functions import *
import matplotlib.pyplot as plt
import numpy as np

def visualize_data_local(data,N):
    # Extract x and y coordinates from the data objects
    x_coords = [instance.coords[0] for instance in data[:-1]]
    y_coords = [instance.coords[1] for instance in data[:-1]]

    # Plot the original data in blue
    plt.scatter(x_coords, y_coords, color='blue')

    # Extract x and y coordinates from the new data instance
    new_x = data[-1].coords[0]
    new_y = data[-1].coords[1]

    # Plot the new data instance in red
    plt.scatter(new_x, new_y, color='red')

    # Extract x and y coordinates from the N set
    N_x_coords = [instance.coords[0] for instance in N]
    N_y_coords = [instance.coords[1] for instance in N]

    plt.scatter(N_x_coords, N_y_coords, color='green')

    # Display the plot
    plt.show()

learning_set = generate_synthetic_data(100)

def LWF(Learning_set, new_instance, kappa, theta, t):
    """
    Parameters:
    Learning_set (list): The current learning set. Each element is a data instance with a 'weight' attribute.
    new_instance (Instance): The new instance to be added to the learning set.
    kappa (int): The number of nearest neighbors to consider.
    theta (float): The threshold. Instances with a weight less than this are removed from the learning set.
    t (int): The decay rate.
    """
    N = nearest_neighbors(Learning_set, new_instance, kappa)
    learning_set.append(new_instance)
    last_neighbor = N[-1]

    for instance in N:
        gamma = gamma_function(new_instance, instance, last_neighbor, t)
        instance.weight = gamma * instance.weight
        if instance.weight < theta:
            learning_set.remove(instance)

    return learning_set,N

def gamma_function(new_instance, instance, last_neighbor, t):

    di2 = euclidean_distance(new_instance, instance)**2
    dk2 = euclidean_distance(new_instance, last_neighbor)**2
    if  di2 > dk2:
        return 1
    else:
        return t + (1 - t) * (di2/dk2) 
    

returned_set,N = LWF(learning_set, DataInstance([6, 2], 'A'), 5, 0.25, 1)
visualize_data_local(returned_set,N)

