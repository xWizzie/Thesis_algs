from General_functions import *

# Initialize lists to store true labels and predicted labels
y_true = []
y_pred = []

# Generate synthetic data
training_data = generate_synthetic_data()

classification = None
cd = list([training_data[0]])
sim_list = {}
# Iterate over each instance x in the training data
for x in training_data[1:]:
    # Initialize the similarity list for this instance
    sim_list = {}
    # Iterate over each instance y in the concept description
    for y in cd:
        # Calculate the similarity between x and y
        sim_list[y] = calc_sim(x, y)
 

    # If the similarity list is not empty 
    if sim_list:
        # Find the instance y in the concept description with the maximum similarity to x
        ymax = max(sim_list, key=sim_list.get)
        
        # Classify x based on the class of ymax
        if x.label == ymax.label:
            classification = 'correct'
        else:
            classification = 'incorrect'
            # Add x to the concept description only if it's classified incorrectly
            cd.append(x)
        print(x.label, " ", ymax.label, " ", classification)
        # Add the true label and predicted label to the respective lists
        y_true.append(x.label)
        y_pred.append(ymax.label)

# Calculate accuracy and precision
acc = accuracy(y_true, y_pred)
prec = precision(y_true, y_pred)

# Print accuracy and precision
print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
