import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm


class IB3:
    def __init__(self, acceptable=0.9, removable=0.75):
        self.acceptable = acceptable
        self.removable = removable
        self.window = None
        self.classes = None
        self.freq_classes = None
        self.knn = None
        self.class_record = None

    def _initialize_knn(self):
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        self.knn.fit(self.window[:, :-1])

    def is_acceptable(self, inst, inst_index):
        """
        Determines if the given instance is acceptable based on the confidence levels of the instance and the class.

        Parameters:
        - inst: The index of the instance.
        - inst_index: The index of the instance in the class record.

        Returns:
        - True if the instance is acceptable, False otherwise.
        """

        n = self.class_record[inst_index][0] + self.class_record[inst_index][1]

        succeses = self.class_record[inst_index][0]
        # The lower bound of the confidence for the the nearest neighbor.
        min_inst = self.min_confidence(succeses, n, self.acceptable)

        n = np.sum(self.freq_classes)
        succeses = self.freq_classes[int(inst)]

        # The lower bound of the confidence for the new instance.
        min_class = self.min_confidence(succeses, n, self.acceptable)

        return min_inst > min_class

    def is_removable(self, inst, inst_index):
        """
        Determines if an instance is removable based on the maximum confidence values.

        Args:
            inst (int): The index of the instance.
            inst_index (int): The index of the instance in the class record.

        Returns:
            bool: True if the instance is removable, False otherwise.
        """
        n = self.class_record[inst_index][0] + self.class_record[inst_index][1]
        succeses = self.class_record[inst_index][0]

        max_inst = self.max_confidence(succeses, n, self.removable)

        n = np.sum(self.freq_classes)
        succeses = self.freq_classes[int(inst)]

        max_class = self.max_confidence(succeses, n, self.removable)

        return max_inst < max_class

    def min_confidence(self, y, n, acceptable):
        """
        Calculates the minimum confidence value based on the given parameters.

        Args:
            y (float): The number of positive instances.
            n (float): The total number of instances.
            conf (float): The confidence level.

        Returns:
            float: The minimum confidence value.
        """
        if n == 0.0:
            return 0
        else:
            frequency = y / n
            acceptable2 = acceptable * acceptable
            n2 = n * n
            val = acceptable * \
                np.sqrt((frequency * (1.0 - frequency) / n) +
                        acceptable2 / (4 * n2))
            numerator = frequency + acceptable2 / (2 * n) - val
            denominator = 1.0 + acceptable2 / n
            return numerator / denominator

    def max_confidence(self, y, n, acceptable):
        """
        Calculates the maximum confidence value based on the given parameters.

        Parameters:
        y (float): The number of positive instances.
        n (float): The total number of instances.
        conf (float): The confidence level.

        Returns:
        float: The maximum confidence value.
        """
        if n == 0.0:
            return 1
        else:
            frequency = y / n
            acceptable2 = acceptable * acceptable
            n2 = n * n
            val = acceptable * \
                np.sqrt((frequency * (1.0 - frequency) / n) +
                        acceptable2 / (4 * n2))
            numerator = frequency + acceptable2 / (2 * n) + val
            denominator = 1.0 + acceptable2 / n
            return numerator / denominator

    def fit(self, X, y):
        self.window = X.copy()  # So that it can be changed
        self.classes = np.unique(y)
        self.freq_classes = np.zeros(len(self.classes))
        self.knn = NearestNeighbors(
            n_neighbors=1, algorithm='ball_tree').fit(X)
        self.class_record = np.zeros((len(X), 2), dtype=int)

        for i, inst_class in enumerate(y):
            inst = X[i]
            index_to_remove = []

            # Step 1: k-NN search

            # Indices of the nearest points in the population matrix.
            _, index = self.knn.kneighbors([inst])
            index = index[0][0]
            nearest_instance = self.window[index]

            # Step 2: Look for the nearest "acceptable"
            if not self.is_acceptable(inst_class, index):
                index = np.random.randint(len(self.window))
                nearest_instance = self.window[index]

            # Step 3: If the class predicted is not correct, add inst as a new concept
            if inst_class != nearest_instance[-1]:
                self.window = np.vstack([self.window, inst])
                self.class_record = np.vstack([self.class_record, [1, 0]])
                self.freq_classes[int(inst_class)] += 1

            # Step 4: Update classification record and remove instances if needed
            distances, indices = self.knn.kneighbors([inst])
            best_distance = distances[0][0]
            for i, d in enumerate(distances[0]):
                if d <= best_distance:
                    if self.window[indices[0][i], -1] == inst_class:
                        self.class_record[indices[0][i], 0] += 1
                    else:
                        self.class_record[indices[0][i], 1] += 1

                    if self.is_removable(inst_class, indices[0][i]):
                        index_to_remove.append(indices[0][i])

            # Remove instances marked
            if index_to_remove:
                self.window = np.delete(self.window, index_to_remove, axis=0)
                self.class_record = np.delete(
                    self.class_record, index_to_remove, axis=0)

    def predict(self, X):
        votes = np.zeros((len(X), len(self.classes)))

        for i, inst in enumerate(X):
            _, index = self.knn.kneighbors([inst])
            index = index[0][0]
            neighbor = self.window[index]
            votes[i, int(neighbor[-1])] += 1

        return np.argmax(votes, axis=1)

