import pandas as pd
import pickle
import scipy.stats

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class PolarisClassifier():
    def __init__(self, data, y_col, n, test_size):
        """
        Constructor function that initializes the class. Each model has its own pipeline with its own set of parameter distributions.

        Arguments:
        - data: a pandas dataframe object with the dependent and independent variables
        - y_col: a string that corresponds to the name of the target variable's column in the data frame
        - n: an int that represents the number of iterations for which to test each model
        - test_size: a float 0 < test_size < 1 that represents the proportion of the testing set
        """

        # initialize the class
        self.data = data
        self.y_col = y_col
        self.pipe = {}
        self.n = n
        self.test_size = test_size

        ### SVM ###
        # store SVM inputs
        self.pipe['svm'] = {}

        # use the standardscaler to scale variables from the dataframe and then apply the SVC classifier
        self.pipe['svm']['steps'] = [('scaler', StandardScaler()), ('SVM', SVC())]
        self.pipe['svm']['pipe'] = Pipeline(self.pipe['svm']['steps'])

        # using RandomSearchCV to determine the optimial parameters, so specify the distributions of the parameters
        self.pipe['svm']['parameters'] = {
            'SVM__C' : scipy.stats.expon(scale = 0.1),
            'SVM__gamma' : scipy.stats.expon(scale = 0.1)
            }

        # create the model object and specify the number of iterations to test
        self.pipe['svm']['model'] = RandomizedSearchCV(self.pipe['svm']['pipe'],
            param_distributions = self.pipe['svm']['parameters'],
            n_iter = self.n,
            verbose = 1)
        
        ### Random Forest ###
        # store RandomForest inputs
        self.pipe['rf'] = {}

        # use the standardscaler to scale variables from the dataframe and then apply the RandomForestClassifier
        self.pipe['rf']['steps'] = [('scaler', StandardScaler()), ('RF', RandomForestClassifier(n_jobs = -1))]
        self.pipe['rf']['pipe'] = Pipeline(self.pipe['rf']['steps'])

        # using RandomSearchCV to determine the optimial parameters, so specify the distributions of the parameters
        self.pipe['rf']['parameters'] = {
            'RF__max_depth' : scipy.stats.randint(low = 1, high = 1000),
            'RF__n_estimators' : scipy.stats.randint(low = 1, high = 1000),
            'RF__min_samples_split' : scipy.stats.randint(low = 2, high = 100),
            'RF__min_samples_leaf' : scipy.stats.randint(low = 1, high = 100),
            'RF__bootstrap' : [True, False]
        }

        # create the model object and specify the number of iterations to test
        self.pipe['rf']['model'] = RandomizedSearchCV(self.pipe['rf']['pipe'],
            param_distributions = self.pipe['rf']['parameters'],
            n_iter = self.n,
            verbose = 1)

        ### KNeighbors ###
        # store RandomForest inputs
        self.pipe['kn'] = {}

        # use the standardscaler to scale variables from the dataframe and then apply the RandomForestClassifier
        self.pipe['kn']['steps'] = [('scaler', StandardScaler()), ('KN', KNeighborsClassifier())]
        self.pipe['kn']['pipe'] = Pipeline(self.pipe['kn']['steps'])

        # using RandomSearchCV to determine the optimial parameters, so specify the distributions of the parameters
        self.pipe['kn']['parameters'] = {
            'KN__n_neighbors' : scipy.stats.randint(low = 1, high = 50),
            'KN__algorithm' : ['ball_tree', 'kd_tree', 'brute'],
            'KN__p' : [1, 2]
        }

        # create the model object and specify the number of iterations to test
        self.pipe['kn']['model'] = RandomizedSearchCV(self.pipe['kn']['pipe'],
            param_distributions = self.pipe['kn']['parameters'],
            n_iter = self.n,
            verbose = 1)

        ### Gradient Boosting ###
        # store RandomForest inputs
        self.pipe['gb'] = {}

        # use the standardscaler to scale variables from the dataframe and then apply the RandomForestClassifier
        self.pipe['gb']['steps'] = [('scaler', StandardScaler()), ('GB', GradientBoostingClassifier())]
        self.pipe['gb']['pipe'] = Pipeline(self.pipe['gb']['steps'])

        # using RandomSearchCV to determine the optimial parameters, so specify the distributions of the parameters
        self.pipe['gb']['parameters'] = {
            'GB__learning_rate' : scipy.stats.expon(scale = 0.1),
            'GB__n_estimators' : scipy.stats.randint(low = 1, high = 1000),
            'GB__criterion' : ['friedman_mse', 'mse', 'mae'],
            'GB__min_samples_split' : scipy.stats.randint(low = 2, high = 100),
            'GB__min_samples_leaf' : scipy.stats.randint(low = 1, high = 100),
            'GB__max_depth' : scipy.stats.randint(low = 1, high = 100)
        }

        # create the model object and specify the number of iterations to test
        self.pipe['gb']['model'] = RandomizedSearchCV(self.pipe['gb']['pipe'],
            param_distributions = self.pipe['gb']['parameters'],
            n_iter = self.n,
            verbose = 1)

        ### Decision Tree ###
        # store RandomForest inputs
        self.pipe['dt'] = {}

        # use the standardscaler to scale variables from the dataframe and then apply the RandomForestClassifier
        self.pipe['dt']['steps'] = [('scaler', StandardScaler()), ('DT', DecisionTreeClassifier())]
        self.pipe['dt']['pipe'] = Pipeline(self.pipe['dt']['steps'])

        # using RandomSearchCV to determine the optimial parameters, so specify the distributions of the parameters
        self.pipe['dt']['parameters'] = {
            'DT__criterion' : ['gini', 'entropy'],
            'DT__splitter' : ['best', 'random'],
            'DT__max_depth' : scipy.stats.randint(low = 1, high = 1000),
            'DT__min_samples_split' : scipy.stats.randint(low = 2, high = 100),
            'DT__min_samples_leaf' : scipy.stats.randint(low = 1, high = 100),
        }

        # create the model object and specify the number of iterations to test
        self.pipe['dt']['model'] = RandomizedSearchCV(self.pipe['dt']['pipe'],
            param_distributions = self.pipe['dt']['parameters'],
            n_iter = self.n,
            verbose = 1)

    def _split(self):
        """
        A function to split the data into training and testing sets.
        Returns x_train, x_test, y_train, y_test as numpy arrays.
        """
        # return train/test split
        return train_test_split(
                self.data.drop(labels = self.y_col, axis = 1),
                self.data[self.y_col],
                test_size = self.test_size)
    
    def fit(self):
        """
        A function that leverages the _split() function to split the data into training and testing sets and then fits each model's pipeline to the data.
        """
        # split data into training and testing sets
        print('Splitting data...')
        self.x_train, self.x_test, self.y_train, self.y_test = self._split()
        print('\nComplete')

        # create a dictionary of model names to print out the status to output
        self.model_names = {
            'svm' : 'SVM',
            'rf' : 'Random Forest',
            'kn' : 'K Neighbors',
            'gb' : 'Gradient Boosting',
            'dt' : 'Decision Tree'
            }
        
        # fit each model to the training set
        for model in self.pipe:
            print(f'\nFitting {self.model_names[model]}...')
            self.pipe[model]['model'].fit(self.x_train, self.y_train)    
    
    def evaluate(self):
        """
        A function that predicts on the testing set and evaluates each model's performance using accuracy, precision, and recall.
        Prints the results to output.
        """
        self.pred = {}
        self.metrics = {}
        for model in self.pipe:
            # predict using the testing set
            self.pred[model] = self.pipe[model]['model'].predict(self.x_test)

            # calculate testing metrics
            self.metrics[model] = {
                'accuracy' : sklearn.metrics.accuracy_score(self.y_test, self.pred[model]),
                'precision' : sklearn.metrics.precision_score(self.y_test, self.pred[model], average = 'weighted'),
                'recall' : sklearn.metrics.recall_score(self.y_test, self.pred[model], average = 'weighted')
            }

            # # print results to output
            # print(self.model_names[model].center(30, '-'))
            # for metric in self.metrics[model]:
            #     print(f'{str(metric.title() + ":").ljust(15, " ")}{str(round(self.metrics[model][metric], 4)).rjust(10, " ")}')
            
            # print('\n')
        
        # create a data frame to output results
        print(pd.DataFrame(data = self.metrics).T.sort_values(by = 'accuracy', axis = 0, ascending = False).apply(lambda x: round(x, 4)))
    
    def get_model(self, model):
        """
        A function that can be called upon to return the model with the best set of parameters.

        Arguments:
        - model: string that corresponds to the different model names ['svm', 'rf', 'kn', 'gb', 'dt']

        Returns a model object.
        """
        return self.pipe[model]['model']
    
    def save_model(self, model, filename = 'model'):
        """
        A function that pickles a selected model to the working directory. Note: if a model with the same name exists, it must be deleted or the name must be changed before pickling again.

        Arguments:
        - model: a string that corresponds to the different model names ['svm', 'rf', 'kn', 'gb', 'dt']
        - filename: a string that represents the name of the output file (ex. 'model')
        """
        # pickle the model object to the working directory
        pickle.dump(self.pipe[model]['model'], open(f'{filename}.p', 'xb'))
