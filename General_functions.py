import matplotlib.pyplot as plt
import numpy as np
import math


def calc_sim(x, y):
    return -math.sqrt(np.sum((np.array(x.coords) - np.array(y.coords))**2))


class DataInstance:
    def __init__(self, coords, label, weight=1):
        self.coords = coords
        self.label = label
        self.classification = ""
        self.weight = weight


def visualize_data(data, cd):
    plt.figure(figsize=(8, 8))
    colors = {'A': 'blue', 'B': 'red'}

    # Plot training data
    for instance in data:
        coords = instance.coords
        plt.scatter(coords[0], coords[1],
                    color=colors[instance.label], marker='o')

    for instance in cd:
        coords = instance.coords
        plt.scatter(coords[0], coords[1],
                    color=colors[instance.label], marker='x')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Synthetic Data Visualization')
    plt.show()


def generate_synthetic_data(num_instances=100):
    np.random.seed(42)  # For reproducibility
    data_objects = []

    for _ in range(num_instances):
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(0, 10)

        # Create a target concept with four disjuncts
        if (x1 < 5 and x2 < 5) or (x1 >= 5 and x2 >= 5):
            label = 'A'
        else:
            label = 'B'

        # Create an instance of DataInstance and append it to the list
        data_instance = DataInstance([x1, x2], label)
        data_objects.append(data_instance)

    return data_objects


def accuracy(y_true, y_pred):
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def precision(y_true, y_pred):
    true_positives = sum(a == b == 1 for a, b in zip(y_true, y_pred))
    total_predicted_positives = sum(pred == 1 for pred in y_pred)
    return true_positives / total_predicted_positives if total_predicted_positives else 0


def nearest_neighbors(Learning_set, new_instance, kappa):
    distances = [(euclidean_distance(instance, new_instance), instance) for instance in Learning_set if instance != new_instance]
    distances.sort(key=lambda x: x[0])
    return [instance for _, instance in distances[:kappa]]

def euclidean_distance(instance1, instance2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(instance1.coords, instance2.coords)))
