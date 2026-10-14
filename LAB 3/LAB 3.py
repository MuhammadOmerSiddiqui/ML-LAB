import math
from collections import Counter
# Node class and prediction (Section 4 and 5)
class Node:
    def __init__(self, node_type, label=None, feature=None):
        self.type = node_type      
        self.label = label         
        self.feature = feature     
        self.branches = {}         
def predict(tree, x):
    current = tree
    while current.type != "leaf":
        feature = current.feature
        value = x[feature]
        current = current.branches[value]
    return current.label
# Activity 3
def activity_3():
    print("ACTIVITY 3 - Implement Prediction")
    fail_leaf = Node("leaf", label="Fail")
    pass_leaf = Node("leaf", label="Pass")
    study_node = Node("internal", feature="Study")
    study_node.branches["Low"] = fail_leaf
    study_node.branches["High"] = pass_leaf
    root = Node("internal", feature="Attendance")
    root.branches["Low"] = fail_leaf
    root.branches["High"] = study_node
    test_students = [
        {"Attendance": "High", "Study": "High"},
        {"Attendance": "High", "Study": "Low"},
        {"Attendance": "Low",  "Study": "High"},
        {"Attendance": "Low",  "Study": "Low"},
    ]
    for s in test_students:
        print(f"Attendance={s['Attendance']:<5} Study={s['Study']:<5} -> {predict(root, s)}")
    print()
# Activity 1
def entropy(labels):
    if len(labels) == 0:
        return 0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    result = 0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)
    return result
def activity_1():
    print("ACTIVITY 1 - Implement Entropy")
    labels1 = ["Pass", "Pass", "Pass", "Pass"]
    labels2 = ["Pass", "Pass", "Fail", "Fail"]
    labels3 = ["Pass", "Fail", "Pass", "Fail"]
    print(f"labels1 {labels1} -> entropy = {entropy(labels1):.4f}")
    print(f"labels2 {labels2} -> entropy = {entropy(labels2):.4f}")
    print(f"labels3 {labels3} -> entropy = {entropy(labels3):.4f}")
    print()
    print("Explanation:")
    print("  labels1: all Pass, no uncertainty -> entropy 0.")
    print("  labels2: 50/50 split -> maximum uncertainty for 2 classes -> entropy 1.")
    print("  labels3: also 50/50 split (order differs, counts identical) -> entropy 1.")
    print("  Entropy depends only on the proportions of each label, not their order.")
    print()
# Dataset (Section 7)
dataset = [
    {"Attendance": "High", "Study": "High", "Assignment": "Yes", "Label": "Pass"},
    {"Attendance": "High", "Study": "Low",  "Assignment": "Yes", "Label": "Pass"},
    {"Attendance": "High", "Study": "Low",  "Assignment": "No",  "Label": "Pass"},
    {"Attendance": "Low",  "Study": "High", "Assignment": "Yes", "Label": "Pass"},
    {"Attendance": "Low",  "Study": "Low",  "Assignment": "No",  "Label": "Fail"},
    {"Attendance": "Low",  "Study": "Low",  "Assignment": "Yes", "Label": "Fail"},
    {"Attendance": "High", "Study": "High", "Assignment": "No",  "Label": "Pass"},
    {"Attendance": "Low",  "Study": "High", "Assignment": "No",  "Label": "Fail"},
]
# Activity 2
def conditional_entropy(dataset, feature, label_col="Label"):
    total = len(dataset)
    weighted_entropy = 0
    values = set(row[feature] for row in dataset)
    for value in values:
        subset = [row[label_col] for row in dataset if row[feature] == value]
        weight = len(subset) / total
        weighted_entropy += weight * entropy(subset)
    return weighted_entropy
def mutual_information(dataset, feature, label_col="Label"):
    labels = [row[label_col] for row in dataset]
    h_y = entropy(labels)
    h_y_given_x = conditional_entropy(dataset, feature, label_col)
    return h_y - h_y_given_x
def activity_2():
    print("ACTIVITY 2 - Implement Mutual Information")
    labels = [row["Label"] for row in dataset]
    print(f"H(Y) = {entropy(labels):.4f}")
    print()
    features = ["Attendance", "Study", "Assignment"]
    for feature in features:
        h_cond = conditional_entropy(dataset, feature)
        mi = mutual_information(dataset, feature)
        print(f"{feature:<11} H(Y|X) = {h_cond:.4f}   MI = {mi:.4f}")
    best = max(features, key=lambda f: mutual_information(dataset, f))
    print(f"\nBest feature (highest MI): {best}")
    print()
# Activity 4
def majority_label(dataset, label_col="Label"):
    labels = [row[label_col] for row in dataset]
    return Counter(labels).most_common(1)[0][0]
def train(dataset, features, label_col="Label"):
    labels = [row[label_col] for row in dataset]
    if len(set(labels)) == 1:
        return Node("leaf", label=labels[0])
    if len(dataset) == 0 or len(features) == 0:
        return Node("leaf", label=majority_label(dataset, label_col) if dataset else "Fail")
    if all(len(set(row[f] for row in dataset)) == 1 for f in features):
        return Node("leaf", label=majority_label(dataset, label_col))
    best_feature = max(features, key=lambda f: mutual_information(dataset, f, label_col))
    node = Node("internal", feature=best_feature)
    remaining = [f for f in features if f != best_feature]
    values = set(row[best_feature] for row in dataset)
    for value in values:
        subset = [row for row in dataset if row[best_feature] == value]
        node.branches[value] = train(subset, remaining, label_col)
    return node
def print_tree(node, indent=0):
    pad = "  " * indent
    if node.type == "leaf":
        print(f"{pad}-> {node.label}")
        return
    print(f"{pad}[{node.feature}]")
    for value, child in node.branches.items():
        print(f"{pad}  {value}:")
        print_tree(child, indent + 2)
def activity_4():
    print("ACTIVITY 4 - Implement Recursive Training")
    global trained_tree
    features = ["Attendance", "Study", "Assignment"]
    trained_tree = train(dataset, features)
    print_tree(trained_tree)
    print()
# Activity 5
def activity_5():
    print("ACTIVITY 5 - Test the Trained Tree")
    new_students = [
        {"Attendance": "High", "Study": "High", "Assignment": "Yes"},
        {"Attendance": "High", "Study": "Low",  "Assignment": "No"},
        {"Attendance": "Low",  "Study": "High", "Assignment": "Yes"},
        {"Attendance": "Low",  "Study": "Low",  "Assignment": "No"},
        {"Attendance": "High", "Study": "Low",  "Assignment": "Yes"},
        {"Attendance": "Low",  "Study": "High", "Assignment": "No"},
    ]
    for i, s in enumerate(new_students, 1):
        print(f"Student {i}: Attendance={s['Attendance']:<5} Study={s['Study']:<5} "
              f"Assignment={s['Assignment']:<4} -> {predict(trained_tree, s)}")
    print()
if __name__ == "__main__":
    activity_1()
    activity_2()
    activity_3()
    activity_4()
    activity_5()
