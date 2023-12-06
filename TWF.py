from General_functions import *
# This algorithm only works for large datasets and does not do much just keep the newest
# instances and discard the rest. The idea is to keep the instances that are most relevant
# to the current concept.


# Weight of each instance is initialized to 1
learning_set = generate_synthetic_data(100)

def TWF(learning_set, new_instance, gamma, theta):
    learning_set.append(new_instance)
    for instance in learning_set:
        instance.weight = gamma * instance.weight
        if instance.weight < theta:
            learning_set.remove(instance)
    return learning_set

TWF(learning_set, DataInstance([5, 3], 'A'), 0.5, 0.25)