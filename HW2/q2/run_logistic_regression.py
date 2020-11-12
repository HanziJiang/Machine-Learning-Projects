from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_logistic_regression():
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    
    hyperparameters = {
        "learning_rate": 0.005,
        "weight_regularization": 0.,
        "num_iterations": 400
    }
    weights = [[0.]] * (M + 1)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    print("\n=============================================================================\n")
    print("The step size is {}.".format(hyperparameters["learning_rate"]))
    print("The number of iterations is: {}.\n".format(hyperparameters["num_iterations"]))
    
    ce_list_train = []
    ce_list_valid = []
    
    for t in range(hyperparameters["num_iterations"]):
        f_train, df_train, y_train = logistic(weights, train_inputs, train_targets, hyperparameters)
        ce_train, frac_correct_train = evaluate(train_targets, y_train)
        ce_list_train.append(ce_train)
        
        f_valid, df_valid, y_valid = logistic(weights, valid_inputs, valid_targets, hyperparameters)
        ce_valid, frac_correct_valid = evaluate(valid_targets, y_valid)
        ce_list_valid.append(ce_valid)
        
        weights = weights - hyperparameters["learning_rate"] * df_train

    ce_train, frac_correct_train = evaluate(train_targets, y_train)
    ce_list_train.append(ce_train)
    print("For the training dataset, the averaged cross entropy is {} and the classification error is {}.".format(ce_train, 1 - frac_correct_train))
    
    f_valid, df_valid, y_valid = logistic(weights, valid_inputs, valid_targets, hyperparameters)
    ce_valid, frac_correct_valid = evaluate(valid_targets, y_valid)
    ce_list_valid.append(ce_valid)
    print("For the validation dataset, the averaged cross entropy is {} and the classification error is {}.".format(ce_valid, 1 - frac_correct_valid))
    
    f_test, df_test, y_test = logistic(weights, test_inputs, test_targets, hyperparameters)
    ce_test, frac_correct_test = evaluate(test_targets, y_test)
    print("For the test dataset, the averaged cross entropy is {} and the classification error is {}.\n".format(ce_test, 1 - frac_correct_test))
    
    plt.plot(range(hyperparameters["num_iterations"] + 1), ce_list_train, label="Training Dataset")
    plt.plot(range(hyperparameters["num_iterations"] + 1), ce_list_valid, label="Validation Dataset")
    plt.legend(loc="upper right")
    plt.xlabel("iteration")
    plt.ylabel("Cross Entropy")
    plt.show()
    print("=============================================================================\n\n")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.004,
        "weight_regularization": 0.,
        "num_iterations": 500
    }
    
    lambd_range = [0.0, 0.001, 0.01, 0.1, 1.0]
    rerun_range = range(5)
    all_ce_train = np.zeros(shape=(len(lambd_range),len(rerun_range) + 2))
    all_error_train = np.zeros(shape=(len(lambd_range),len(rerun_range) + 2))
    all_ce_valid = np.zeros(shape=(len(lambd_range),len(rerun_range) + 2))
    all_error_valid = np.zeros(shape=(len(lambd_range),len(rerun_range) + 2))
    
    print("\n=============================================================================\n")
    print("The step size is {}.".format(hyperparameters["learning_rate"]))
    print("The number of iterations is: {}.\n".format(hyperparameters["num_iterations"]))
    
    for i in range(len(lambd_range)):
        hyperparameters["weight_regularization"] = lambd_range[i]
        all_ce_train[i][0] = lambd_range[i]
        all_ce_valid[i][0] = lambd_range[i]
        all_error_train[i][0] = lambd_range[i]
        all_error_valid[i][0] = lambd_range[i]
        
        for j in rerun_range:
            weights = [[0.]] * (M + 1)
            
            if j == 0:
                ce_list_train = []
                ce_list_valid = []
            
            for t in range(hyperparameters["num_iterations"]):
                f, df, y = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                
                if j == 0:
                    y_train = logistic_predict(weights, train_inputs)
                    ce_train, frac_correct_train = evaluate(train_targets, y_train)
                    ce_list_train.append(ce_train)
                    y_valid = logistic_predict(weights, valid_inputs)
                    ce_valid, frac_correct_valid = evaluate(valid_targets, y_valid)
                    ce_list_valid.append(ce_valid)
                    
                weights = weights - hyperparameters["learning_rate"] * df
            
            if j == 0:
                y_train = logistic_predict(weights, train_inputs)
                ce_train, frac_correct_train = evaluate(train_targets, y_train)
                ce_list_train.append(ce_train)
                
                y_valid = logistic_predict(weights, valid_inputs)
                ce_valid, frac_correct_valid = evaluate(valid_targets, y_valid)
                ce_list_valid.append(ce_valid)
                
                print("\n==================================================")
                print("Plot for lambda={}.".format(hyperparameters["weight_regularization"]))
                print("The CE is {} for training and {} for validation.".format(ce_list_train[hyperparameters["num_iterations"]], ce_list_valid[hyperparameters["num_iterations"]]))
                plt.plot(range(hyperparameters["num_iterations"] + 1), ce_list_train, label="Training Dataset")
                plt.plot(range(hyperparameters["num_iterations"] + 1), ce_list_valid, label="Validation Dataset")
                plt.legend(loc="upper right")
                plt.xlabel("iteration")
                plt.ylabel("Cross Entropy")
                plt.show()
                
                # Uncomment to see test results only once you have decided lambda
#                 y_test = logistic_predict(weights, test_inputs)
#                 ce_test, frac_correct_test = evaluate(test_targets, y_test)
#                 print("The CE is {} and the classification error is {} for test.".format(ce_test, 1-frac_correct_test))
            
            y = logistic_predict(weights, train_inputs)
            all_ce_train[i][j+1], all_error_train[i][j+1] = evaluate(train_targets, y)
            all_error_train[i][j+1] = 1 - all_error_train[i][j+1]
            
            y = logistic_predict(weights, valid_inputs)
            all_ce_valid[i][j+1], all_error_valid[i][j+1] = evaluate(valid_targets, y)
            all_error_valid[i][j+1] = 1 - all_error_valid[i][j+1]
            
        all_ce_train[i][-1] = np.average(all_ce_train[i][1:-1])
        all_ce_valid[i][-1] = np.average(all_ce_valid[i][1:-1])
        all_error_train[i][-1] = np.average(all_error_train[i][1:-1])
        all_error_valid[i][-1] = np.average(all_error_valid[i][1:-1])
    
    print("\nCE results for training data:")
    df_ce_train = pd.DataFrame(data=all_ce_train, columns=["lambda", "run 0", "run 1", "run 2", "run 3", "run 4", "average"])
    print(df_ce_train.to_string(index=False))
    
    print("\nClassification errors for training data:")
    df_error_train = pd.DataFrame(data=all_error_train, columns=["lambda", "run 0", "run 1", "run 2", "run 3", "run 4", "average"])
    print(df_error_train.to_string(index=False))
    
    print("\nCE results for validation data:")
    df_ce_valid = pd.DataFrame(data=all_ce_valid, columns=["lambda", "run 0", "run 1", "run 2", "run 3", "run 4", "average"])
    print(df_ce_valid.to_string(index=False))
    
    print("\nClassification errors for validation data:")
    df_error_valid = pd.DataFrame(data=all_error_valid, columns=["lambda", "run 0", "run 1", "run 2", "run 3", "run 4", "average"])
    print(df_error_valid.to_string(index=False))
    
    lambd = 0.001
    print()
    
    print("\n=============================================================================\n\n")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)
    
    y, dy, = logistic(weights, data, targets,hyperparameters)[:2]

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    #run_logistic_regression()
    run_pen_logistic_regression()