import dis
from posixpath import split
from pyexpat import features
from tkinter.ttk import LabeledScale
from turtle import pos
import numpy as np
from collections import defaultdict
from collections import Counter
from numpy.random import default_rng
import pickle

class cls_node():
    
    def __init__(self, feature_idx, best_guess, entropy, split_values, node_type):
        self.feature_idx = feature_idx
        self.best_guess = best_guess
        self.entropy = entropy
        self.split_values = split_values
        self.node_type = node_type
        self.parents = dict()
        self.children = dict()

    def add_node(self, idx, node):
        node.parents[len(self.parents)] = (self.feature_idx, self.split_values[idx])
        self.children[idx] = node

    def print_node(self):
        print(f"feature: {self.feature_idx}" )
        print(f"best guess: {self.best_guess}")
        print(f"entropy: {self.entropy}")
        print(f"split values {self.split_values}")
        print(f"type: {self.node_type}")
        for child in self.children:
            if type(self.children[child]) == cls_node:
                print(f"child value: {self.split_values[child]}    child branch feature: {self.children[child].feature_idx}")
            else:
                print(f"child value: {self.split_values[child]}    child leaf: {self.children[child]}")

class custom_decision_tree():


    def __init__(self, features, labels, continious_size = 10, discrete_idx_list = [], min_samples = 2):
        self.feature_matrix = features
        self.label_vector = labels
        self.continious_size = continious_size
        self.discrete_idx_list = discrete_idx_list
        self.split_values = []
        self.min_samples = min_samples
        self.labaled_idx = 0
        self.progress = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        self.total_labels = len(labels)

    def calc_entropy(self,idx_list):
        temp_labels = [self.label_vector[idx] for idx in idx_list]
        total = len(temp_labels)
        if 0 not in temp_labels or 1 not in temp_labels:
            entropy = 0
            best_guess = int(1 in temp_labels) # we know that temp labels is all zeros or all ones, if 1 in temp labels it is true so it should be 1 else it should be 0
        else:
            positive_prob = sum(temp_labels)/total
            negative_prob = 1 - positive_prob
            entropy = (-negative_prob * np.log2(negative_prob)) + (-positive_prob * np.log2(positive_prob))
            best_guess = int(positive_prob >= negative_prob) # if positive prob is equal or higher than negative prob guess 1
        
        return entropy, best_guess
    
    # https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8
    def discrete_gain(self, feature_idx, idx_list):
        data_list = [self.feature_matrix[idx,feature_idx] for idx in idx_list]
        entropy = self.calc_entropy(idx_list)[0]
        categories = Counter(data_list)
        weighted_entropy = 0
        for category in categories:
            temp_idx_list = [idx_list[i] for i, value in enumerate(data_list) if value == category]
            weighted_entropy += categories[category]/len(data_list) * self.calc_entropy(temp_idx_list)[0]

        Info_gain = entropy - weighted_entropy
        split_values = list(set(data_list))
        return Info_gain, split_values

    def continious_gain(self, feature_idx, idx_list):
        data_list = [self.feature_matrix[idx,feature_idx] for idx in idx_list]
        entropy = self.calc_entropy(idx_list)[0]
        
        best_weighted_entropy = entropy
        # print(len(idx_list))
        index_list_1 = []
        index_list_2 = idx_list.copy()
        total = len(data_list)
        data_values = sorted(set(data_list))
        split_value = data_values[0]
        for j, value in enumerate(data_values):
            temp_index_list = [idx_list[i] for i in range(len(data_list)) if data_list[i] == value]
            index_list_1 += temp_index_list
            index_list_2 = [idx for idx in index_list_2 if idx not in temp_index_list]
            weighted_entropy = len(index_list_1)/total * self.calc_entropy(index_list_1)[0] + len(index_list_2)/total * self.calc_entropy(index_list_2)[0]
            if weighted_entropy >= best_weighted_entropy:
                continue
            else:
                best_weighted_entropy = weighted_entropy
                if j != len(list(enumerate(data_values)))-1:
                    split_value = (value+data_values[j+1])/2
        
        total_uniques_values = len(data_values)
        denominator = 2
        current_point = total_uniques_values / denominator
        current_value = total_uniques_values[current_point]
        denominator *= 2
        index_list_left = [idx_list[i] for i in range(len(data_list)) if data_list[i] <= current_value]
        index_list_right = [idx_list[i] for i in range(len(data_list)) if data_list[i] > current_value]
        current_entropy = len(index_list_left)/total * self.calc_entropy(index_list_left)[0] + len(index_list_right)/total * self.calc_entropy(index_list_right)[0]
        
        diff = total_uniques_values / denominator
        pos_value = total_uniques_values[current_point+diff]
        neg_value = total_uniques_values[current_point-diff]
        index_list_left = [idx_list[i] for i in range(len(data_list)) if data_list[i] <= pos_value]
        index_list_right = [idx_list[i] for i in range(len(data_list)) if data_list[i] > pos_value]
        pos_entropy = len(index_list_left)/total * self.calc_entropy(index_list_left)[0] + len(index_list_right)/total * self.calc_entropy(index_list_right)[0]
        
        index_list_left = [idx_list[i] for i in range(len(data_list)) if data_list[i] <= neg_value]
        index_list_right = [idx_list[i] for i in range(len(data_list)) if data_list[i] > neg_value]
        neg_entropy = len(index_list_left)/total * self.calc_entropy(index_list_left)[0] + len(index_list_right)/total * self.calc_entropy(index_list_right)[0]
        while diff > 2:    
            denominator *= 2
            diff = total_uniques_values /denominator
            if neg_entropy > pos_entropy:
                pos_value = current_value
                current_value = total_uniques_values[current_point - diff]
                pos_entropy = current_entropy
            else:
                neg_value = current_value
                current_value = total_uniques_values[current_point + diff]
                neg_entropy = current_entropy
            index_list_left = [idx_list[i] for i in range(len(data_list)) if data_list[i] <= current_value]
            index_list_right = [idx_list[i] for i in range(len(data_list)) if data_list[i] > current_value]
            current_entropy = len(index_list_left)/total * self.calc_entropy(index_list_left)[0] + len(index_list_right)/total * self.calc_entropy(index_list_right)[0]

        
        Info_gain = entropy - best_weighted_entropy
        split_values = [(-np.inf,split_value),(split_value,np.inf)]
        return Info_gain, split_values

    def find_best_split(self, idx_list, feature_idx_list = []):
        best_idx = 0
        best_ig = 0
        for feature_idx in range(self.feature_matrix.shape[1]):
            if feature_idx in feature_idx_list:
                continue
            if feature_idx in self.discrete_idx_list:
                Info_gain = self.discrete_gain(feature_idx, idx_list)
                type = "discrete"
            else:
                Info_gain = self.continious_gain(feature_idx, idx_list)
                type = "continious"
            if Info_gain[0] > best_ig:
                best_ig = Info_gain[0]
                best_values = Info_gain[1]
                best_idx = feature_idx
                best_type = type
                print(best_ig)
                
            
        return best_idx, best_ig, best_type, best_values
    

    def build_node(self, idx_list, node, feature_idx_list = []):
        feature_idx = node.feature_idx
        for i, split_value in enumerate(node.split_values):
            if feature_idx in self.discrete_idx_list:
                temp_idx_list = [idx for idx in idx_list if self.feature_matrix[idx,feature_idx] == split_value]
            else:
                temp_idx_list = [idx for idx in idx_list if self.feature_matrix[idx,feature_idx] >= split_value[0] and self.feature_matrix[idx,feature_idx] < split_value[1]]
            try:
                positve_prob = sum([self.label_vector[idx] for idx in temp_idx_list])/len(temp_idx_list)
                negative_prob = 1 - positve_prob
                entropy, best_guess = self.calc_entropy(temp_idx_list)
            except:
                node.print_node()
                print(split_value)
                print(temp_idx_list)
                temp_list = [self.feature_matrix[idx,feature_idx] for idx in idx_list]
                print(temp_list)
            if best_guess:
                prob = positve_prob
            else:
                prob = negative_prob
           
            if len(temp_idx_list) < self.min_samples:    
                node.children[i] = {"best_guess" : best_guess, "prob": prob}
                self.labaled_idx += len(temp_idx_list)
            elif entropy == 0:
                node.children[i] = {"best_guess": best_guess, "prob": prob}
                self.labaled_idx += len(temp_idx_list)
            else:
                temp_idx, Info_gain, type, split_values = self.find_best_split(temp_idx_list,feature_idx_list)
                if temp_idx == 1:
                    print(split_values)
                temp_node = cls_node(temp_idx, best_guess, entropy, split_values, type)
                temp_feature_idx_list = feature_idx_list.copy()
                temp_feature_idx_list.append(temp_idx)
                temp_node = self.build_node(temp_idx_list.copy(),temp_node, temp_feature_idx_list.copy())
                node.children[i] = temp_node
            if self.labaled_idx/self.total_labels > self.progress[0]:
                print(self.progress[0])
                self.progress.pop(0)
        return node
    
    def build_tree(self):
        idx_list = [i for i in range(len(self.feature_matrix))]
        best_idx, Info_gain, type, split_values = self.find_best_split(idx_list)
        entropy, best_guess = self.calc_entropy(idx_list)
        self.tree = cls_node(best_idx, best_guess, entropy, split_values, type)
        self.tree = self.build_node(idx_list.copy(), self.tree, [best_idx].copy())
                


# C:\Users\karst\OneDrive\Documenten\uni\jaar 3 semester 2, het semester dat ik opgaf\Bachelor-graduation-project\code\notebooks\feature_matrix.npy
map_path = "code/notebooks/"
feature_matrix = np.load(map_path + "feature_matrix.npy")
label_matrix = np.load(map_path + "label_matrix.npy")
sentence_idx = list(np.load(map_path + "sentence_idx.npy"))

total_sentences = len(set(sentence_idx))
split = 0.5
rng = default_rng()
training_sentence_ids = rng.choice(total_sentences, size=int(total_sentences*split), replace=False)
test_sentence_ids = [i for i in range(total_sentences) if i not in training_sentence_ids]

training_id_list = []
test_id_list = []
for i, item in enumerate(sentence_idx):
    if item in training_sentence_ids:
        training_id_list.append(i)
    else:
        test_id_list.append(i)

training_feature_matrix = np.array([feature_matrix[i] for i in training_id_list])
training_label_matrix = np.array([label_matrix[i] for i in training_id_list])
test_feature_matrix = np.array([feature_matrix[i] for i in test_id_list])
test_label_matrix = np.array([label_matrix[i] for i in test_id_list])


test_array = np.array([
    [0,-3,5,2,1],
    [0,-2,6,2,0],
    [0,-1,7,2,1],
    [0,0,4,1,0],
    [1,1,3,1,1],
    [1,2,2,1,0],
    [1,3,8,0,1],
    [1,4,1,0,0],
])
test_array_labels = np.array([0,0,1,1,0,0,1,1])

# test = custom_decision_tree(training_feature_matrix, training_label_matrix, discrete_idx_list=[-2,-3,-4,-5,-6])
test = custom_decision_tree(test_array,test_array_labels,discrete_idx_list=[0,3,4])
test.build_tree()
tree = test.tree
tree.print_node()

# with open(map_path + 'custom_tree.pkl', 'wb') as outp:
#     pickle.dump(tree, outp)

# features = np.array([[0,0,0],[1,2,0],[3,4,0],[3,1,1],[4,3,1],[5,5,1]])

# test = [1,1,1,1,2,2,2,2,2,0,0]
# print(list(set(test)))


# labels = np.array([0,0,0,1,1,1])
# test = custom_decision_tree(features, labels, discrete_idx_list=[-1])
# idx_list = [0,1,2,3,4,5]
# # print(test.calc_entropy(idx_list))
# # print(test.continious_gain(0,idx_list))
# # print(test.continious_gain(1,idx_list))
# # print(test.find_best_split(idx_list))