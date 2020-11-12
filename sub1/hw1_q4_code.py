import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SEED = 98

def shuffle_data(data):
    idx = np.random.permutation(len(data["X"]))
    X, t = data["X"][idx], data["t"][idx]
    return {"X": X, "t": t}

def split_data(data, num_folds, fold):
    N = len(data["X"])
    chuck_size = N / float(num_folds)
    start_index = round(chuck_size * (fold - 1))
    end_index = round(chuck_size * fold)
    data_fold = {"X": data["X"][start_index:end_index], 
                 "t": data["t"][start_index:end_index]}
    data_rest = {"X": np.concatenate((data["X"][:start_index], data["X"][end_index:])), 
                 "t": np.concatenate((data["t"][:start_index], data["t"][end_index:]))}
    return data_fold, data_rest

def train_model(data, lambd):
    N = len(data["X"])
    D = len(data["X"][0])
    model = np.dot(np.transpose(data["X"]), data["t"])
    model = np.matmul(np.linalg.inv(np.matmul(np.transpose(data["X"]), data["X"]) + lambd * N * np.identity(D)), model)
    return model
    
def predict(data, model):
    return np.matmul(data["X"], model)

def loss(data, model):
    return np.linalg.norm(predict(data, model) - data["t"], ord=2)**2/(2 * len(data["X"]))
    
def cross_validation(data, num_folds, lambd_seq):
    np.random.seed(SEED)
    data = shuffle_data(data)
    cv_error = [0] * len(lambd_seq)
    for i in range(0, len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds + 1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = cv_loss_lmd / num_folds
    return cv_error
    
if __name__ == "__main__":
    lambd_seq = np.linspace(start=0.00005, stop=0.005, num=50)
    
    data_train = {"X": np.genfromtxt("./data/data_train_X.csv", delimiter=",", autostrip=True), 
                  "t": np.genfromtxt("./data/data_train_y.csv", delimiter=",", autostrip=True)}
    data_test = {"X": np.genfromtxt("./data/data_test_X.csv", delimiter=",", autostrip=True), 
                 "t": np.genfromtxt("./data/data_test_y.csv", delimiter=",", autostrip=True)}    
    
    cv_error_5 = cross_validation(data_train, 5, lambd_seq)
    cv_error_10 = cross_validation(data_train, 10, lambd_seq)
    
    cv_errors_df = pd.DataFrame({'lambda': lambd_seq, 'CV Error 5-fold': cv_error_5, 'CV Error 10-fold': cv_error_10})
    print(cv_errors_df.to_string(index=False))
    
    
    indices = np.where(cv_error_5 == np.amin(cv_error_5))[0]
    print("\nThe proposed lambda by 5-fold is {}.".format(lambd_seq[indices]))
    indices = np.where(cv_error_10 == np.amin(cv_error_10))[0]
    print("The proposed lambda by 10-fold is {}.".format(lambd_seq[indices]))
    
    print("\n====================================================\n")
    
    loss_train = [0] * len(lambd_seq)
    loss_test = [0] * len(lambd_seq)
    
    for i in range(0, len(lambd_seq)):
        lambd = lambd_seq[i]
        model = train_model(data_train, lambd)
        loss_train[i] = loss(data_train, model)
        loss_test[i] = loss(data_test, model)
        
    errors_df = pd.DataFrame({'lambda': lambd_seq, 'Training Error': loss_train, 'Testing Error': loss_test})
    print (errors_df.to_string(index=False))
    
    plt.plot(lambd_seq, loss_train, label="Training Error")
    plt.plot(lambd_seq, loss_test, label="Test Error")
    plt.plot(lambd_seq, cv_error_5, label="CV Error 5-fold")
    plt.plot(lambd_seq, cv_error_10, label="CV Error 10-fold")
    plt.legend(loc="upper right")
    plt.xlabel("lambda")
    plt.ylabel("Error")
    plt.show()
    
    print("\n====================================================\n")
    
    print("As lambda increases, the training error increases. This is because there is a tradeoff between the fit to the data and the magnitude of the weights. As we penalize weight values more with larger lambda, the fit to the data decreases, resulting in a larger training error.")
    print("As lambda increases, the test error, CV error 5-fold and the CV error 10-fold first decrease then increase. When lambda is extremely small, we do not penalize large weights very much, so our model might overfit the training data. As we increase lambda, our test result will be better. However, if the lambda is too large, the test error will be large because we have preferred simpler model too much, that we fail to capture important properties.")
    print("The CV error estimates the test error well. The shapes of the test error and the CV errors are similar. The lambdas proposed by 5-fold CV and 10-fold CV are similar (or the same).")
    