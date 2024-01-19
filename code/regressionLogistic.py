import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay


class MultinomialLogisticRegression:
    def __init__(self, alpha=0.001, epochs=1000):

        self.alpha = alpha
        self.epochs = epochs


    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


    def fit(self, X, y):
        X = self.add_intercept(X)
        self.w = np.random.rand(X.shape[1], y.shape[1])

        for i in range(self.epochs):
            z = np.dot(X, self.w)
            y_pred = self.softmax(z)
            gradient = np.dot(X.T, (y_pred - y))
            self.w -= self.alpha * gradient

    def predict_probabilidades(self, X):
        X = self.add_intercept(X)
        result = self.softmax(np.dot(X, self.w))
        return result

    def predict(self, X):
        proba = self.predict_probabilidades(X)
        return np.argmax(proba, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def plot_roc_curve(self, X, y):
        y_pred = self.predict_probabilidades(X)
        fpr, tpr, thresholds = roc_curve(y, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()


    def plot_roc_curve_multiclass(self, X, y, name_model,  index_in_for):
        y_pred = self.predict_probabilidades(X)
        fpr_0, tpr_0, thresholds = roc_curve(y == 0, y_pred[:, 0])
        fpr_1, tpr_1, thresholds_1 = roc_curve(y == 1, y_pred[:, 1])
        fpr_2, tpr_2, thresholds_2 = roc_curve(y == 2, y_pred[:, 2])
        roc_auc_0 = auc(fpr_0, tpr_0)
        roc_auc_1 = auc(fpr_1, tpr_1)
        roc_auc_2 = auc(fpr_2, tpr_2)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr_0, tpr_0, label=f'ROC curve of class Normal  (area = {roc_auc_0:.2f})')
        plt.plot(fpr_1, tpr_1, label=f'ROC curve of class Sospechoso (area = {roc_auc_1:.2f})')
        plt.plot(fpr_2, tpr_2, label=f'ROC curve of class Sospechoso (area = {roc_auc_2:.2f})')
        plt.plot()
        plt.title(f'{name_model} number iteration {index_in_for}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

    def plot_precision_recall_curve(self, X, y):
        y_pred = self.predict_probabilidades(X)
        precision, recall, thresholds = precision_recall_curve(y, y_pred[:, 1])
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower right')
        plt.show()



# K-Fold Cross Validation
def kfold_cv(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    precisions = []
    recalls = []
    f1_scores = []
    acc_scores = []

    for i,(train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print(f'y_train es:{type(y_train)}, y shape es {np.shape(y_train)}')
        model.fit(X_train, np.eye(3)[y_train])
        y_pred = model.predict(X_test)


        # Calcular las métricas de evaluación del modelo
        precision, recall, f1, acc = evaluate_model(y_test, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        acc_scores.append(acc)

        #ROC Curve kfol cross validation
        model.plot_roc_curve_multiclass(X_train, y_train, 'Regression Logistic - K_Fold Cross Validation', i )

        #confusion matrix
        cm = confusion_matrix(y_test,y_pred)
        cm_display =ConfusionMatrixDisplay(cm).plot()
        plt.title(f'Confusion Matrix - Regression Logistic - K Fold Cross Validation- iter {i}')
        plt.show()


    print(f"K-Fold Cross Validation Accuracy: {np.mean(acc_scores):.2f}")
    print(f"K-Fold Cross Validation Precision: {np.mean(precisions):.2f}")
    print(f"K-Fold Cross Validation Recall: {np.mean(recalls):.2f}")
    print(f"K-Fold Cross Validation F1-Score: {np.mean(f1_scores):.2f}")

    print (classification_report(y_test, y_pred))
    print (confusion_matrix(y_test, y_pred))
    

    return model

# Bootstrap
def bootstrap(X, y, model, n_bootstraps=100):
    precisions = []
    recalls = []
    f1_scores = []
    acc_scores = []

    rs = ShuffleSplit(n_splits=n_bootstraps, test_size=0.2)
    for i,(train_idx, test_idx) in enumerate(rs.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model.fit(X_train, np.eye(3)[y_train])
        y_pred = model.predict(X_test)

        # Calcular las métricas de evaluación del modelo
        precision, recall, f1, acc = evaluate_model(y_test, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        acc_scores.append(acc)

        #roc curve bootstrap
        model.plot_roc_curve_multiclass(X_train, y_train, 'Regression Logistic - Bootstrapping ', i )
        
                #confusion matrix
        cm = confusion_matrix(y_test,y_pred)
        cm_display =ConfusionMatrixDisplay(cm).plot()
        plt.title(f'Confusion Matrix - Bootstrap - K Fold Cross Validation- iter {i}')
        plt.show()


    print(f"Bootstrap Accuracy: {np.mean(acc_scores):.2f}")
    print(f"Bootstrap Precision: {np.mean(precisions):.2f}")
    print(f"Bootstrap Recall: {np.mean(recalls):.2f}")
    print(f"Bootstrap F1-Score: {np.mean(f1_scores):.2f}")

    print (classification_report(y_test, y_pred))
    print (confusion_matrix(y_test, y_pred))
    
    return model

#metrics
def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return precision, recall, f1, acc


#display bar data
def proporcions_dataset(count_label_1 , count_label_2 , count_label_3):
        plt.bar(
        ["class 1(%d)"%count_label_1, "class 2(%d)"%count_label_2, "class 3(%d)"%count_label_3],
        [count_label_1, count_label_2, count_label_3],
        color=["#ff0033", "#1120ff", "#22aa66"],
        width= 0.8)
        plt.show()



if __name__ == '__main__':

    #import dataset
    dataset = pd.read_csv('data\DataSet_CTG_train.csv')

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

    label_1 = label_1.sample(n = min_label, replace=False)
    label_2 = label_2.sample(n=min_label, replace=False)
    label_3 = label_3.sample(n=min_label, replace=True)
    
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

    # [5] train and validation
    model = MultinomialLogisticRegression()

    model_kfold = kfold_cv(XAll, yAll, model,n_splits=3)
    model_bootstrap = bootstrap(XAll, yAll, model,n_bootstraps=5)




    import warnings
    warnings.filterwarnings("ignore")

    # [6] test with test data
    data_test = pd.read_csv('data\DataSet_CTG_test.csv')

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



