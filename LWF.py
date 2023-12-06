from General_functions import *

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

    for instance in N[:-1]:
        gamma = gamma_function(new_instance, instance, last_neighbor, t)
        instance.weight = gamma * instance.weight
        if instance.weight < theta:
            learning_set.remove(instance)

    return learning_set

def gamma_function(new_instance, instance, last_neighbor, t):

    di2 = euclidean_distance(new_instance, instance)**2
    dk2 = euclidean_distance(new_instance, last_neighbor)**2
    if  di2 > dk2:
        return 1
    else:
        return t + (1 - t) * (di2/dk2) 
    

visualize_data(learning_set,LWF(learning_set, DataInstance([5, 3], 'A'), 3, 0.25, 1))