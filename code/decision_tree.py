
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler,  RobustScaler
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



# Creating a decision tree
class Node:
    def __init__(self, data, target):
        self.left = None
        self.right = None
        self.data = data
        self.target = target
        self.feature_split = None
        self.threshold_split = None
        self.is_leaf = False
        self.prediction = None

class DecisionTree:
    def __init__(self, max_depth=5):
        self.root = None
        self.max_depth = max_depth

    def _entropy(self, data):
        _, counts = np.unique(data, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p))

    def _information_gain(self, data, feature, threshold):
        entropy_parent = self._entropy(data)

        left_indices = np.where(feature <= threshold)[0]
        right_indices = np.where(feature > threshold)[0]

        n_left = len(left_indices)
        n_right = len(right_indices)

        if n_left == 0 or n_right == 0:
            return 0

        entropy_left = self._entropy(data[left_indices])
        entropy_right = self._entropy(data[right_indices])

        weighted_entropy = (n_left / len(data)) * entropy_left + (n_right / len(data)) * entropy_right

        return entropy_parent - weighted_entropy

    def _get_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_info_gain = -np.inf

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            feature = X[:, feature_idx]
            thresholds = np.unique(feature)

            for threshold in thresholds:
                info_gain = self._information_gain(y, feature, threshold)

                if info_gain > best_info_gain:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_info_gain = info_gain

        return best_feature, best_threshold, best_info_gain

    def _build_tree(self, X, y, current_depth=0):
        n_samples, n_features = X.shape

        if current_depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < 2:
            leaf_node = Node(data=X, target=y)
            leaf_node.is_leaf = True
            leaf_node.prediction = np.bincount(y).argmax()
            return leaf_node

        best_feature, best_threshold, best_info_gain = self._get_best_split(X, y)

        if best_info_gain == 0:
            leaf_node = Node(data=X, target=y)
            leaf_node.is_leaf = True
            leaf_node.prediction = np.bincount(y).argmax()
            return leaf_node

        node = Node(data=X, target=y)
        node.feature_split = best_feature
        node.threshold_split = best_threshold

        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]

        node.left = self._build_tree(X[left_indices], y[left_indices], current_depth + 1)
        node.right = self._build_tree(X[right_indices], y[right_indices], current_depth + 1)

        return node

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.prediction

        if x[node.feature_split] <= node.threshold_split:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])



# K-Fold Cross Validation
def kfold_cv(X, y, DecisionTree, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    precisions = []
    recalls = []
    f1_scores = []
    acc_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Entrenar el modelo
        DecisionTree.fit(X_train, y_train)

        # Predecir con el modelo
        y_pred = DecisionTree.predict(X_test)

        # Calcular las métricas de evaluación del modelo
        precision, recall, f1, acc = evaluate_model(y_test, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        acc_scores.append(acc)

    print(f"K-Fold Cross Validation Accuracy: {np.mean(acc_scores):.2f}")
    print(f"K-Fold Cross Validation Precision: {np.mean(precisions):.2f}")
    print(f"K-Fold Cross Validation Recall: {np.mean(recalls):.2f}")
    print(f"K-Fold Cross Validation F1-Score: {np.mean(f1_scores):.2f}")

    print (classification_report(y_test, y_pred))
    print (confusion_matrix(y_test, y_pred))

    return DecisionTree

# Bootstrap
def bootstrap(X, y, DecisionTree , n_bootstraps=100):
    precisions = []
    recalls = []
    f1_scores = []
    acc_scores = []

    rs = ShuffleSplit(n_splits=n_bootstraps, test_size=0.2, random_state=42)
    for train_idx, test_idx in rs.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Entrenar el modelo
        DecisionTree.fit(X_train, y_train)

        # Predecir con el modelo
        y_pred = DecisionTree.predict(X_test)

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
    return DecisionTree

def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return precision, recall, f1, acc

def proporcions_dataset(count_label_1 , count_label_2 , count_label_3):
        plt.bar(
        ["normal(%d)"%count_label_1, "sospechoso(%d)"%count_label_2, "patologico(%d)"%count_label_3],
        [count_label_1, count_label_2, count_label_3],
        color=["#ff0033", "#1120ff", "#22aa66"],
        width= 0.8)
        plt.show()


if __name__ == '__main__':
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
    scaler = RobustScaler()
    XAll = data.iloc[:, :-1].values
    XAll = scaler.fit_transform(XAll)
    yAll = data.iloc[:, -1].values
    yAll = yAll.astype(int)

    # [4] change classes to 0,1,2
    classes = {'1': 0, '2': 1, '3': 2}

    # [5] train and validate with kfold_cv and bootstrap
    DecisionTree = DecisionTree()

    # [5.1]  use the kfold_cv
    DecisionTree_kfold_cv = kfold_cv(XAll, yAll, DecisionTree)

    # [5.2]  use the bootstrap
    DecisionTree_bootstrap = bootstrap(XAll, yAll, DecisionTree)

    # [6] test with test data
    data_test = pd.read_csv('D:\Documentos\MISCURSOS\Proyectos Git Hub\Cardiotocografia_fetal-MachineLearning\data\DataSet_CTG_test.csv')
    data_test = data_test.dropna()
    XTest = data_test.values
    XTest = scaler.transform(XTest)

    # [6.1] predict with kfold_cv
    y_pred_kfold = DecisionTree_kfold_cv.predict(XTest)
    y_pred_kfold = y_pred_kfold.astype(int)

    #save y_pred_kfold into a csv file
    np.savetxt("./y_pred_kfold_decisiontree.csv", y_pred_kfold, delimiter=",")

    # [6.2] predict with bootstrap
    y_pred_bootstrap = DecisionTree_bootstrap.predict(XTest)
    y_pred_bootstrap = y_pred_bootstrap.astype(int)

    #save y_pred_bootstrap into a csv file
    np.savetxt("./y_pred_bootstrap_decisiontree.csv", y_pred_bootstrap, delimiter=",")


    # print (classification_report(y_test, y_pred_kfold_cv))




