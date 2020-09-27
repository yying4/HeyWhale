import numpy as np

'''
Calculate the prediction yi^ by f(b+w*xi), 
if b+w*xi >= 0, yi^ = 1;
if b+w*xi < 0, yi^ = 0.
'''
def predict(x, w):
    x = x.reshape(x.shape[0], 1)  # input
    w = w.reshape(1, w.shape[0])  # weight
    return 1 if w.dot(x)[0][0] >= 0 else 0



# Train the data using Perceptron
def training(data):
    # The parameter of bias is equal to 1，
    # because the predicted label yi^ = f(b+w*xi).
    bias = np.ones((len(data), 1))

    # add 1 as the first input for each record
    data = np.concatenate((bias, data), 1)  

    learning_rate = 1
    weight = np.zeros(data.shape[1] - 1)  # set initial weights as 0s

    iteration_time = 0
    correct_count = 0 
    while correct_count < data.shape[0]:
        correct_count = 0
        for item in data:
            yi = item[-1]  # label-yi
            xi = item[:-1]  # data-xi
            yi_hat = predict(xi, weight)

            # update wj <- wj + learn_rate*(yi-yi^)*xi
            weight = weight + learning_rate * (yi - yi_hat) * xi
            if yi == yi_hat: correct_count += 1
        iteration_time += 1
        print("Correct times in iteration {}: {}".format(iteration_time, correct_count))

    print("Weight after training: " + str(weight))
    return weight



# Predict labels using weight from training
def testing(data, weight):
    bias = np.ones((len(data), 1))

    data = np.concatenate((bias, data), 1)  
    for i in data:
        xi = i
        yi_hat = predict(xi, weight)
        print(yi_hat)


'''
Training data:
records consist of xi1, xi2 and yi, 
xis are the features for training
'''
train_data = np.array([
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
    ], dtype=np.int)

weight = training(train_data)


'''
Testing data:
records consist of only xis without labels
'''
test_data = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype=np.int)

print('\nPredictions:')
testing(test_data, weight)
