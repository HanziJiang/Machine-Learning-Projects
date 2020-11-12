from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(real_path, fake_path):
    """ Load, preprocess and split the dataset.
    """
    with open(real_path) as data:
        real_data = data.read().splitlines()
    with open(fake_path) as data:
        fake_data = data.read().splitlines()
    
    vectorizer = CountVectorizer()
    vec = vectorizer.fit_transform(real_data + fake_data)

    targets = [True for i in range(len(real_data))] + [False for i in range(len(fake_data))]

    vec_train, vec_test, tar_train, tar_test = train_test_split(vec, targets, test_size=0.3, random_state=11)
    vec_test, vec_validate, tar_test, tar_validate = train_test_split(vec_test, tar_test, test_size=0.5, random_state=11)
    return vec_train, vec_test, vec_validate, tar_train, tar_test, tar_validate

def select_knn_model(vec_train, vec_validate, tar_train, tar_validate, vec_test, tar_test, ks, metric=None):
    """ Use a KNN classifer to classify between real and fake news.
    """
    train_errors, validate_errors, train_accuracy, validate_accuracy = [], [], [], []
    best_k, best_accuracy = 0, 0
    
    for k in ks:
        if metric is None:
            model = KNeighborsClassifier(n_neighbors=k) 
        else:
            model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(vec_train, tar_train)
        
        # training accuracy
        accuracy = model.score(vec_train, tar_train)
        train_accuracy.append(accuracy)
        train_errors.append(1 - accuracy)
        
        # validation accuracy
        accuracy = model.score(vec_validate, tar_validate)
        validate_accuracy.append(accuracy)
        validate_errors.append(1 - accuracy)
    
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    print("====================================================")
    print("Metric is default.\n")
    
    errors_df = pd.DataFrame({'k': ks, 'Training Error': train_errors, 'Validation Error': validate_errors})
    print (errors_df.to_string(index=False))
    
    plt.plot(ks, train_accuracy, label="Training Dataset")
    plt.plot(ks, validate_accuracy, label="Validation Dataset")
    plt.legend(loc="upper right")
    plt.xlabel("k")
    plt.xticks(np.arange(min(ks), max(ks)+1, step=1.0))
    plt.ylabel("Accuracy")
    plt.show()
    
    print("The k that results in the best model is {}.\nThe validation accuracy for this best model is {}.".format(best_k, best_accuracy))
    model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    model.fit(vec_train, tar_train)
    accuracy = model.score(vec_test, tar_test)
    print("The test accuracy is {}.\n\n".format(accuracy))
  

if __name__ == "__main__":
    vec_train, vec_test, vec_validate, tar_train, tar_test, tar_validate = load_data("./data/clean_real.txt", "./data/clean_fake.txt")
  
    ks = range(1, 21)
    
    select_knn_model(vec_train, vec_validate, tar_train, tar_validate, vec_test, tar_test, ks, 'minkowski')
    select_knn_model(vec_train, vec_validate, tar_train, tar_validate, vec_test, tar_test, ks, 'cosine')
    print('''The cosine metric calculates the cosine angle between two vectors to determine their similarity. Two vectors in the same direction has a cosine similarity of 1, in opposite direction -1, perpendicular to each other 0. It measures how similar two vectors irrespective of their sizes. ''')
    print('''The accuracy increases when we change the metric from default (minikowski) to cosine. This is because a heading may contain a word more than once. In fact, if a heading is longer, it is more likely to contain a word more than once. When classifying headings into true or false, we want to see what words are in the heading and their distributions, as opposed to the length of the heading. Take ["computer", "computer computer computer", "time"] as an example. It is obvious "computer" and "computer computer computer" are more similar than "computer" and "time" or "computer computer computer" and "time". "computer" and "computer computer computer" are on the same axis - the "computer" axis. However, the Euclidean distance between "computer" and "computer computer computer" is larger because the latter is very long. The distance between them is 2 units, in the "computer" axis. Even though "computer" and "time" are completely different and are on different axis, their Euclidean distance is smaller (sqrt((1 unit in "computer" axis)^2 + (1 unit in "time" axis)^2)), therefore will be falsely determined as more similar by the minkowski metric. If we use the cosine metric, the angle between the "computer" vector and the "computer computer computer" vector is 1, larger than that between "computer" and "time" or "computer computer computer" and time, which is 0 because they are oriented perpendicular to each other. Therefore, we should really use cosine similarity when we care about the orientation of vectors, not the magnitude. In otherwords, the model overfits the training data if lambda is too small, underfits the data if lambda is too large. 
''')