import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics import precision_score


class Node():
    def __init__(self, leaf=True, label=None, splitting_feature_id=None, splitting_treshold=None, left_child=None, right_child=None, info_gain = None):
   
        self.label = label
        self.leaf = leaf
        self.splitting_feature_id = splitting_feature_id
        self.splitting_treshold = splitting_treshold
        self.left_child = left_child
        self.right_child = right_child
        self.info_gain = info_gain


class BinaryTreeClassifier():

    def __init__(self, max_depth=10, prune=False, criterion="gini", max_features="sqrt",  random_state=None, class_weights=[], verbose=0, max_thresholds=None, min_sample_split=2):
        """Create a Binary Tree Classifier.

        Parameters
        ----------
        max_depth : int, default=10
            The maximum depth of the tree.
        prune : bool, default=False
            Whether to perform pruning after training.
        criterion : {"gini", "entropy", "log_loss"}, default="gini"
            The function to measure the quality of a split.
        max_features : {"sqrt", "log2"} or int or float, default="sqrt"
            The number of features to consider when looking for the best split.
        random_state : int or None, default=None
            The seed used by the random number generator.
        verbose : int, default=0
            Controls the verbosity of the tree-building process.
        max_thresholds : int or None, default=None
            The maximum number of thresholds to consider when looking for the best split.
        """

        self.tree = None
        self.max_depth = max_depth
        self.prune = prune
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose
        self.max_thresholds = max_thresholds
        self.min_sample_split = min_sample_split
        random.seed(self.random_state)
        self.unique_values = []
        self.unique_counts = []
        self.class_weights = class_weights

    def calculate_error(self, predictions, labels):
        # Calculate the error of the predictions
        wrong_predictions = np.sum(predictions != labels)
        error = wrong_predictions / len(labels)
        return error

    def accuracy(self, predictions, labels):
        correct_predictions = np.sum(np.fromiter(
            (1 if predictions[i] == labels[i] else 0 for i in range(len(predictions))), dtype=np.int32))
        return correct_predictions / len(labels)

    def precision(self, predictions, labels):

        tp = np.sum(np.fromiter((1 if predictions[i] == labels[i] else 0 for i in range(
            len(predictions))), dtype=np.int32))
        fp = np.sum(np.fromiter((1 if predictions[i] == 1 and labels[i] == 0 else 0 for i in range(
            len(predictions))), dtype=np.int32))
        return (tp / (tp + fp))

    def recall(self, predictions, labels):
        # Calculate the recall of the predictions
        tp = np.sum(np.fromiter((1 if predictions[i] == labels[i] else 0 for i in range(
            len(predictions))), dtype=np.int32))
        fn = np.sum(np.fromiter((1 if predictions[i] == 0 and labels[i] == 1 else 0 for i in range(
            len(predictions))), dtype=np.int32))
        return tp / (tp + fn)

    def f1_score(self, predictions, labels):
        # Calculate the f1-score of the predictions
        prec = self.precision(predictions, labels)
        rec = self.recall(predictions, labels)
        return 2 * (prec * rec) / (prec + rec)

    def __filter_df(self, X, condition):
        """Filter the DataFrame X based on a condition.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame to filter.
        condition : tuple (int, float)
            The condition used for filtering.

        Returns
        -------
        left_X : pandas DataFrame
            The filtered DataFrame containing rows where the condition is met.
        left_y : pandas DataFrame
            The labels corresponding to the filtered rows in left_X.
        right_X : pandas DataFrame
            The filtered DataFrame containing rows where the condition is not met.
        right_y : pandas DataFrame
            The labels corresponding to the filtered rows in right_X.
        """

        X_conditioned = X.iloc[:, condition[0]]

        left_child = X[X_conditioned <= float(condition[1])]
        right_child = X[X_conditioned > float(condition[1])]
        return left_child.iloc[:, : -1], left_child.iloc[:, -1], right_child.iloc[:, : -1], right_child.iloc[:, -1]

    def __pruning_result(self, tree, y_train, X_val, y_val):
        """Perform pruning and return the pruned tree or a leaf label.

        Parameters
        ----------
        tree : Node or int
            The tree (or leaf label) to evaluate for pruning.
        y_train : pandas DataFrame
            The training labels.
        X_val : pandas DataFrame
            The validation features.
        y_val : pandas DataFrame
            The validation labels.

        Returns
        -------
        Node or int
            The pruned tree or a leaf label based on the pruning decision.
        """

        leaf = y_train.value_counts().index[0]
        errors_leaf = sum(y_val != leaf)
        errors_decision_node = sum(y_val != self.predict(X_val))

        if errors_leaf <= errors_decision_node:

            return leaf
        else:
            return tree

    def __post_pruning(self, tree, X_train, y_train, X_val, y_val):
        """Recursively prune the tree using validation data.

        Parameters
        ----------
        tree : Node or int
            The tree (or leaf label) to evaluate and potentially prune.
        X_train : pandas DataFrame
            The training features.
        y_train : pandas DataFrame
            The training labels.
        X_val : pandas DataFrame
            The validation features.
        y_val : pandas DataFrame
            The validation labels.

        Returns
        -------
        Node or int
            The pruned tree or a leaf label.
        """

        condition = [x for x in tree.keys()][0]
        left_child, right_child = tree[condition]

        if not isinstance(left_child, dict) and not isinstance(right_child, dict):
            return self.__pruning_result(tree, y_train, X_val, y_val)

        else:
            X_train_left, y_train_left, X_train_right, y_train_right = self.__filter_df(
                pd.concat([X_train, y_train], axis=1), condition)
            X_val_left, y_val_left, X_val_right, y_val_right = self.__filter_df(
                pd.concat([X_val, y_val], axis=1), condition)

            if isinstance(left_child, dict):
                left_child = self.__post_pruning(
                    left_child, X_train_left, y_train_left, X_val_left, y_val_left)

            if isinstance(right_child, dict):
                right_child = self.__post_pruning(
                    right_child, X_train_right, y_train_right, X_val_right, y_val_right)

            if left_child == right_child:
                return left_child

            new_tree = {condition: [left_child, right_child]}

            return new_tree

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        if self.verbose:
            print(f"Input features type: {type(X_train)}")
            print(f"Input labels type: {type(y_train)}")

        X_df = pd.DataFrame(X_train)
        y_df = pd.DataFrame(y_train)

        if self.verbose:
            print(f"Generating tree")

        self.tree = self.__create_tree_classifier(X_df, y_df)

        if self.prune:
            X_val_df = pd.DataFrame(X_val)
            y_val_df = pd.DataFrame(y_val)
            self.tree = self.from_dict(
                self.__post_pruning(self.tree_dict(), X_df, y_df, X_val_df, y_val_df))

    def __calculate_entropy(self, column_values: np.ndarray) -> float:
        unique_values, value_counts = np.unique(column_values, return_counts=True)
        probabilities = value_counts / len(column_values)

        if len(self.class_weights) > 0 and len(unique_values) > 0:
            default_weight = self.class_weights.get(unique_values[0], 1.0)
            probabilities *= default_weight

        entropy_criteria = {
            'shannon': lambda probs: -np.sum(probs * np.log2(probs)),
            'scaled': lambda probs: -np.sum((probs / 2) * np.log2(probs)),
            'gini': lambda probs: 1 - np.sum(np.square(probs))
        }

        entropy_function = entropy_criteria.get(self.criterion, entropy_criteria['shannon'])
        entropy_value = entropy_function(probabilities)

        return entropy_value

    def __best_split(self, node_features: pd.DataFrame, node_labels: pd.DataFrame, parent_entropy: float) -> tuple:
        best_threshold = None
        best_feature_id = None
        best_info_gain = -np.inf
        n_features = len(node_features.columns)
        node_labels_len = len(node_labels)

        columns_id = []

        if self.max_features is not None:
            if self.max_features == "sqrt":
                columns_id = np.random.choice(range(n_features), int(math.sqrt(n_features)), replace=False)
            elif self.max_features == "log2":
                columns_id = np.random.choice(range(n_features), int(math.log2(n_features)), replace=False)
            elif isinstance(self.max_features, int):
                if self.max_features > n_features:
                    raise ValueError("Max features > number of features")
                elif self.max_features <= 0:
                    raise ValueError("Max features must be > 0")
                columns_id = np.random.choice(range(n_features), self.max_features, replace=False)
            elif isinstance(self.max_features, float):
                if self.max_features > 1:
                    raise ValueError("Max features > number of features")
                elif self.max_features <= 0:
                    raise ValueError("Max features must be > 0")
                columns_id = np.random.choice(range(n_features), int(n_features * self.max_features), replace=False)
        else:
            columns_id = list(range(n_features))

        for id in columns_id:
            if self.max_thresholds is not None:
                if self.max_thresholds <= 0:
                    raise ValueError("max_thresholds must be > 0")

                thresholds = np.percentile(node_features.iloc[:, id], np.linspace(0, 100, self.max_thresholds))
            else:
                unique_vals = node_features.loc[:, id].unique()
                thresholds = (unique_vals[1:] + unique_vals[:-1]) / 2

            feature_id = node_features.loc[:, id]

            for threshold in thresholds:
                left_labels = node_labels.loc[feature_id <= threshold]
                right_labels = node_labels.loc[feature_id > threshold]

                left_side_entropy = self.__calculate_entropy(left_labels)
                right_side_entropy = self.__calculate_entropy(right_labels)

                w_left_side_entropy = (len(left_labels) / node_labels_len) * left_side_entropy
                w_right_side_entropy = (len(right_labels) / node_labels_len) * right_side_entropy

                info_gain = parent_entropy - (w_left_side_entropy + w_right_side_entropy)

                if info_gain > best_info_gain:
                    best_threshold = threshold
                    best_feature_id = id
                    best_info_gain = info_gain

        return best_feature_id, best_threshold, best_info_gain

    def __create_tree_classifier(self, X: pd.DataFrame, y: pd.DataFrame, depth: int = 0) -> Node:
        node = Node()

        unique_labels = np.unique(y)
        num_unique_labels = len(unique_labels)
        max_depth_reached = (self.max_depth is not None and depth >= self.max_depth)
        insufficient_samples = len(y) < self.min_sample_split

        if max_depth_reached or num_unique_labels <= 1 or insufficient_samples:
            node.label = np.max(y)
            return node

        best_feature_id, best_treshold, info_gain = self.__best_split(X, y, self.__calculate_entropy(y))

        if info_gain == 0 or best_treshold is None:
            node.label = np.max(y)
            return node
        else:
            node.splitting_feature_id = best_feature_id
            node.splitting_treshold = best_treshold
            node.leaf = False

        X_best_features = X.iloc[:, best_feature_id]
        is_left = X_best_features <= best_treshold

        left_X = X[is_left]
        right_X = X[~is_left]

        left_y = y[is_left]
        right_y = y[~is_left]

        node.left_child = self.__create_tree_classifier(left_X, left_y, depth=depth + 1)
        node.right_child = self.__create_tree_classifier(right_X, right_y, depth=depth + 1)

        left_child_label = node.left_child.label
        right_child_label = node.right_child.label

        if left_child_label is not None and left_child_label == right_child_label:
            node.splitting_feature_id = None
            node.splitting_treshold = None
            node.leaf = True
            node.label = left_child_label

        return node

    def __depth_tree(self, row):

        current_node = self.tree
        while (True):
            if current_node.leaf:
                return current_node.label
            else:
                feature_id = current_node.splitting_feature_id
                threshold = current_node.splitting_treshold
                if (row[feature_id] <= threshold):
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

    def predict(self, X):
        """Predict class labels for a set of samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict class labels for.

        Returns
        -------
        list of int
            The predicted class labels.
        """
        X_train = pd.DataFrame(X)
        return list(X_train.apply(self.__depth_tree, axis=1))

    def __recursive_print(self, node: Node, indent=""):
        """Recursively print the binary tree.

        Parameters
        ----------
        node : Node
            The current node to be printed.
        indent : str, default=""
            The indentation string for formatting the tree.
        """
        if node == None:
            return
        if (node.leaf == True):
            print("{}leaf - label: {} ".format(indent, node.label))
            return

        print("{}id:{} - threshold: {}".format(indent,
              node.splitting_feature_id, node.splitting_treshold))

        self.__recursive_print(node.left_child, "{}   ".format(indent))
        self.__recursive_print(node.right_child, "{}   ".format(indent))

    def show_tree(self):
        """Display the binary tree structure."""
        self.__recursive_print(self.tree)

    def from_dict(self, n):
        """Reconstruct a binary tree from a nested dictionary representation.

        Parameters
        ----------
        n : dict or int
            The nested dictionary representation or leaf label.

        Returns
        -------
        Node
            The root node of the reconstructed binary tree.
        """
        if not isinstance(n, dict):
            return Node(label=n)
        else:
            condition = [k for k in n.keys()][0]
            t = n[condition][0]
            f = n[condition][1]
            left_child = self.from_dict(t)
            right_child = self.from_dict(f)

            return Node(leaf=False, splitting_feature_id=condition[0],
                        splitting_treshold=condition[1], left_child=left_child, right_child=right_child)

    def tree_dict(self):
        """Get a nested dictionary representation of the binary tree.

        Returns
        -------
        dict
            A nested dictionary representation of the binary tree.
        """

        def recursive_get(node: Node):
            if (node.leaf):
                return node.label

            return {(node.splitting_feature_id, node.splitting_treshold): [recursive_get(node.left_child), recursive_get(node.right_child)]}

        return recursive_get(self.tree)

    def score(self, X, y):
        """Compute the precision score for the model's predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The true labels.

        Returns
        -------
        float
            The precision score of the model's predictions.
        """

        y_pred = self.predict(X)

        print(f"Precision: {self.precision(y_pred,y)}")
        print(f"Recall: {self.recall(y_pred,y)}")
        print(f"Accuracy: {self.accuracy(y_pred,y)}")
        print(f"Error: {self.calculate_error(y_pred,y)}")
        print(f"F1-score: {self.f1_score(y_pred,y)}")

        return precision_score(self.predict(X), y)

    def get_params(self, deep=True):
        # Return the hyperparameters of the estimator as a dictionary
        return {
            'max_depth': self.max_depth,
            'prune': self.prune,
            'criterion': self.criterion,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'max_thresholds': self.max_thresholds
        }

    def set_params(self, **params):
        """Get the hyperparameters of the estimator as a dictionary.

        Parameters
        ----------
        deep : bool, default=True
            Whether to include nested objects.

        Returns
        -------
        dict
            A dictionary containing the hyperparameters of the estimator.
        """
        # Set the hyperparameters of the estimator using a dictionary
        for param, value in params.items():
            setattr(self, param, value)
        return self 