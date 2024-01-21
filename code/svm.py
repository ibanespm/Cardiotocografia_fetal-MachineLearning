import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from  sklearn.metrics import ConfusionMatrixDisplay


# K-Fold Cross Validation
def kfold_cv(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    precisions = []
    recalls = []
    f1_scores = []
    acc_scores = []

    for i,(train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Entrenar el modelo - kernel lineal
        clf = SVC(kernel='poly', decision_function_shape='ovr')

        # Train the classifier on the training set
        clf.fit(X_train, y_train)

        # Make predictions on the testing settgb
        y_pred = clf.predict(X_test)

        # Calcular las métricas de evaluación del modelo
        precision, recall, f1, acc = evaluate_model(y_test, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        acc_scores.append(acc)
        

        #confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_display= ConfusionMatrixDisplay(cm).plot()
        plt.title(f'SVM-Confusion Matrix-Kfold Cross Validation-numero de fold {i +1}')
        plt.show()

    print(f"K-Fold Cross Validation Accuracy: {np.mean(acc_scores):.2f}")
    print(f"K-Fold Cross Validation Precision: {np.mean(precisions):.2f}")
    print(f"K-Fold Cross Validation Recall: {np.mean(recalls):.2f}")
    print(f"K-Fold Cross Validation F1-Score: {np.mean(f1_scores):.2f}")

    print (classification_report(y_test, y_pred))
    print (confusion_matrix(y_test, y_pred))

    return clf

# Bootstrap
def bootstrap(X, y, n_bootstraps=100):
    precisions = []
    recalls = []
    f1_scores = []
    acc_scores = []

    rs = ShuffleSplit(n_splits=n_bootstraps, test_size=0.2, random_state=42)
    for train_idx, test_idx in rs.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Entrenar el modelo - kernel lineal
        clf = SVC(kernel='linear', decision_function_shape='ovr')

        # Train the classifier on the training set
        clf.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = clf.predict(X_test)

        # Calcular las métricas de evaluación del modelo
        precision, recall, f1, acc = evaluate_model(y_test, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        acc_scores.append(acc)

    print(f"Bootstrap Accuracy: {np.mean(acc_scores):.2f}")
    print(f"Bootstrap Precision: {np.mean(precisions):.2f}")
    print(f"Bootstrap Recall: {np.mean(recalls):.2f}")
    print(f"Bootstrap F1-Score: {np.mean(f1_scores):.2f}")

    print (classification_report(y_test, y_pred))
    print (confusion_matrix(y_test, y_pred))
    return clf



def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return precision, recall, f1, acc

#proporcion de la data
def proporcions_dataset(count_label1, count_label2, count_label3):
    
    plt.bar(
            ["normal(%d)"%count_label1,"sospechoso(%d)"%count_label2,"patalogico(%d)"%count_label3],
            [ count_label1, count_label2,  count_label3],
            color=["#00ff00", "#777700", "#00ffaa"],
            width=0.8
            )
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('D:\Documentos\MISCURSOS\Proyectos Git Hub\Cardiotocografia_fetal-MachineLearning\data\DataSet_CTG_train.csv')

    # [1] homologar frecuencias de clases
    label_1_count = len(data[data['CLASE'] == 1])
    label_2_count = len(data[data['CLASE'] == 2])
    label_3_count = len(data[data['CLASE'] == 3])



    print(f'count label 1:{label_1_count}, count label 2:{label_2_count}, count label 3:{label_3_count}')
    #proporcion de la data
    proporcions_dataset(label_1_count,label_2_count,label_3_count)



    #label_min_count = min(label_1_count, label_2_count, label_3_count)
    
    #sobremuestro dataset
    number_samples=280

    label_1_sobremuestreado = data[data['CLASE'] == 1].sample(n=number_samples, replace=False)
    label_2_sobremuestreado = data[data['CLASE'] == 2].sample(n=number_samples, replace=False)
    label_3_sobremuestreado = data[data['CLASE'] == 3].sample(n=number_samples, replace= True)

    #nuevo proporcion de dataset
    label_1_count_new = len(label_1_sobremuestreado)
    label_2_count_new = len(label_2_sobremuestreado)
    label_3_count_new = len(label_3_sobremuestreado)

    print(f'Data sobremuestreado : count label 1:{label_1_count_new}, count label 2:{label_2_count_new}, count label 3:{label_3_count_new}')

    proporcions_dataset(label_1_count_new,  label_2_count_new,  label_3_count_new)


    # [2] remove nan
    data = data.dropna()

    # [3] use scaler
    scaler = MinMaxScaler()
    XAll = data.iloc[:, :-1].values
    XAll = scaler.fit_transform(XAll)
    yAll = data.iloc[:, -1].values
    yAll = yAll.astype(int)



    # [5] train and validate with kfold_cv and bootstrap

    # [5.1]  use the kfold_cv
    SVM_kfold = kfold_cv(XAll, yAll, n_splits=3)

    # [5.2]  use the bootstrap
    SVM_bootstrap = bootstrap(XAll, yAll, n_bootstraps=100)

    # [6] test with test data
    data_test = pd.read_csv('D:\Documentos\MISCURSOS\Proyectos Git Hub\Cardiotocografia_fetal-MachineLearning\data\DataSet_CTG_test.csv')
    data_test = data_test.dropna()
    XTest = data_test.values
    XTest = scaler.transform(XTest)

    # [6.1] predict with kfold_cv
    y_pred_kfold = SVM_kfold.predict(XTest)
    y_pred_kfold = y_pred_kfold.astype(int)

    # save y_pred_kfold into a csv file
    np.savetxt("./y_pred_kfold_SVM.csv", y_pred_kfold, delimiter=",")

    # [6.2] predict with bootstrap
    y_pred_bootstrap = SVM_bootstrap.predict(XTest)
    y_pred_bootstrap = y_pred_bootstrap.astype(int)

    # save y_pred_bootstrap into a csv file
    np.savetxt("./y_pred_bootstrap_SVM.csv", y_pred_bootstrap, delimiter=",")

    import warnings
    warnings.filterwarnings("ignore")



