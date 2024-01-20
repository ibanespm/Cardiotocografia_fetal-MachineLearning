import pandas as pd
import numpy as np
from  sklearn.neighbors import KDTree
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, RobustScaler, MinMaxScaler
import time 
import matplotlib.pyplot as plt


class KNN_Linear:
    """Tiene una complejidad temporal de O(n), es decir, que en miles o millones para predecir puede demorar mucho"""

    def __init__(self, k = 3):
        
        self.k = k
    
    def euclidean_distance(self, point_main, point_normal):
        
        distance = np.sqrt(np.sum((point_main- point_normal)**2))
        return distance 

    def fit(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []

        for i in range(x_test.shape[0]):
            distances = np.array([self.euclidean_distance(x_test[i], x) for x in self.x_train])
            indices = np.argpartition(distances, self.k)[:self.k]
            k_nearest_labels = self.y_train[indices]

            # Realiza la predicción tomando la etiqueta más común entre los k vecinos más cercanos
            prediction = np.bincount(k_nearest_labels.astype(int)).argmax()
            predictions.append(prediction)

        return np.array(predictions)



#class KNN_KDTree:
class KDTree_KNN:
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance 

    def fit(self, X, y):
        self.y = y
        self.tree = KDTree(X ,  metric=self.distance)

    def predict(self, x):
        predictiones = []

        for xi in x:
            _, indexes = self.tree.query(xi.reshape(1, -1), k=self.k)  
            y_labels = self.y[indexes]  

            y_labels = y_labels.astype(int)
            y_pred = np.array(
                [np.argmax(np.bincount(labels)) for labels in y_labels])  # predecir la etiqueta de la instancia
            predictiones.append(y_pred[0])
        return np.array(predictiones)



#KFol Cross Validation
def KNN_KFold_cv(X, y, model, num_splits = 4):

    kf_cv =  KFold(n_splits=num_splits, shuffle= True, random_state= 42)
    ratio_error = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracy_scores = []

    for i, (train_index, test_index) in enumerate(kf_cv.split(X)):
        x_train , y_train = X[train_index], y[train_index]
        x_test, y_test = X[test_index], y[test_index]
        #Entrenar el modelo

        model.fit(x_train, y_train)


        #Prediction
        y_prediction = model.predict(x_test)

        ratio_error.append(np.mean(y_prediction !=y_test ))

        #calcular metricas del modelo
        precision, recall, f1, accuracy = evaluate_model(y_test, y_prediction)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    print(f"K-Fold Cross Validation Accuracy: {np.mean(accuracy_scores):.2f}")
    print(f"K-Fold Cross Validation Precision: {np.mean(precisions):.2f}")
    print(f"K-Fold Cross Validation Recall: {np.mean(recalls):.2f}")
    print(f"K-Fold Cross Validation F1-Score: {np.mean(f1_scores):.2f}")

    print (classification_report(y_test, y_prediction))
    print (confusion_matrix(y_test, y_prediction))


    return model, ratio_error


# Bootstrap
def KNN_bootstrap(X, y, model, n_bootstraps=100):

    ratio_error = []
    precisions = []
    recalls = []
    f1_scores = []
    acc_scores = []

    rs = ShuffleSplit(n_splits=n_bootstraps, test_size=0.2, random_state=42)
    for i,(train_idx, test_idx) in enumerate(rs.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        ratio_error.append(np.mean(y_pred !=y_test ))


        # Calcular las métricas de evaluación del modelo
        precision, recall, f1, acc = evaluate_model(y_test, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        acc_scores.append(acc)

        #roc curve bootstrap

    print(f"Bootstrap Accuracy: {np.mean(acc_scores):.2f}")
    print(f"Bootstrap Precision: {np.mean(precisions):.2f}")
    print(f"Bootstrap Recall: {np.mean(recalls):.2f}")
    print(f"Bootstrap F1-Score: {np.mean(f1_scores):.2f}")

    print (classification_report(y_test, y_pred))
    print (confusion_matrix(y_test, y_pred))
    
    return model, ratio_error


def evaluate_model(y_test, y_prediction):
    precision = precision_score(y_test, y_prediction, average='macro')
    recall  = recall_score (y_test, y_prediction, average='macro')
    f1 = f1_score(y_test, y_prediction, average='macro')
    accuracy = accuracy_score (y_test, y_prediction)

    return precision, recall, f1, accuracy


#display bar data
def proporcions_dataset(count_label_1 , count_label_2 , count_label_3):
        plt.bar(
        ["normal(%d)"%count_label_1, "sospechoso(%d)"%count_label_2, "patologico(%d)"%count_label_3],
        [count_label_1, count_label_2, count_label_3],
        color=["#ff0033", "#1120ff", "#22aa66"],
        width= 0.8)
        plt.show()





def trains_models (number):

    for i in range(1, number):
            
            # [5] train and validation
            model = KNN_Linear(k = i)
            print(f'-------------------ITERACION- K  {i }----------------')
            model_kfold, ratio_error_kfold  = KNN_KFold_cv(XAll, yAll, model,num_splits=3)
            model_bootstrap, ratio_error_bootstrat = KNN_bootstrap(XAll, yAll, model,n_bootstraps=5)

    return model_kfold, model_bootstrap, ratio_error_kfold, ratio_error_bootstrat





def show_best_k(ratio_error, name):
    print(f"{name} - Ratio Error: {ratio_error}")

    plt.figure(figsize=(6, 5))
    plt.plot(range(1,  len(ratio_error)+ 1), ratio_error, color="#00ff88", linestyle="dashed", marker='o',
            markerfacecolor='#ff1200')
    plt.xticks(range(1, len(ratio_error)+1))
    plt.title(f'{name} -- Ratio Error vs k')
    plt.xlabel('k')
    plt.ylabel('Ratio Error')
    plt.show()




if __name__ == "__main__":

    #dataset
    dataset = pd.read_csv('D:\Documentos\MISCURSOS\Proyectos Git Hub\Cardiotocografia_fetal-MachineLearning\data\DataSet_CTG_train.csv')
    
    label_1 = dataset[dataset['CLASE'] ==1]
    label_2 = dataset[dataset['CLASE'] ==2]
    label_3 = dataset[dataset['CLASE'] ==3]

    #count labels dataset
    count_label_1 = len(dataset[dataset['CLASE'] ==1])
    count_label_2 = len(dataset[dataset['CLASE'] ==2])
    count_label_3 = len(dataset[dataset['CLASE'] ==3])

    print(f'label 1:{count_label_1}, label 2:{count_label_2}, label 3:{count_label_3}')

    #proporcion de los datos (desequilibrado)
    proporcions_dataset(count_label_1 , count_label_2 , count_label_3)

    #sobremuestreo de la data
    min_label = 280

    label_1 = label_1.sample(n = min_label, replace=False, random_state= 42)
    label_2 = label_2.sample(n=min_label, replace=False, random_state= 42)
    label_3 = label_3.sample(n=min_label, replace=True, random_state= 42)
    
    data = pd.concat([label_1, label_2, label_3])

    count_label_1_oversampled = len(data[data['CLASE'] ==1])
    count_label_2_oversampled = len(data[data['CLASE'] ==2])
    count_label_3_oversampled = len(data[data['CLASE'] ==3])
    
    #new proporsion de los datos(equilibrado)
    proporcions_dataset(count_label_1_oversampled , count_label_2_oversampled , count_label_3_oversampled)      

    # [2] remove nan
    data = data.dropna()

    # [3] use scaler
    scaler = MinMaxScaler()
    XAll = data.iloc[:, :-1].values
    XAll = scaler.fit_transform(XAll)
    yAll = data.iloc[:, -1].values
    yAll = yAll.astype(int)

    # [4] change classes to 0,1,2
    classes = {'1': 0, '2': 1, '3': 2}
    yAll = yAll-1
    



    #kFoldCrossValidation - Bootstraping --- find best k 
    model_kfold, model_bootstrap, ratio_error_kfold, ratio_error_bootstrat = trains_models(30)
    

    print(f'Tenemos len: {np.shape(ratio_error_kfold)}')
    print(f'Tenemos len: {np.shape(ratio_error_bootstrat)}') 

    show_best_k(ratio_error_kfold, 'KNN- KFold')
    show_best_k(ratio_error_bootstrat, 'KNN-Bootstrap')

    import warnings
    warnings.filterwarnings("ignore")

    # [6] test with test data
    data_test = pd.read_csv('D:\Documentos\MISCURSOS\Proyectos Git Hub\Cardiotocografia_fetal-MachineLearning\data\DataSet_CTG_test.csv')

    data_test = data_test.dropna()
    XTest = data_test.values
    XTest = scaler.transform(XTest)


    y_pred_kfold = model_kfold.predict(XTest)
    y_pred_kfold = y_pred_kfold + 1
    y_pred_kfold = y_pred_kfold.astype(int)

    #save y_pred_kfold into a csv file
    np.savetxt("./y_pred_kfold_logistic.csv", y_pred_kfold, delimiter=",")

    y_pred_bootstrap = model_bootstrap.predict(XTest)
    y_pred_bootstrap = y_pred_bootstrap + 1
    y_pred_bootstrap = y_pred_bootstrap.astype(int)
    np.savetxt("./y_pred_bootstrap_logistic.csv", y_pred_bootstrap, delimiter=",")






