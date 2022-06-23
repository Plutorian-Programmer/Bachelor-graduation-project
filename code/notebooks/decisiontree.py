from cProfile import label
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
import math

class cls_forest():
    def __init__(self):
        self.forest = []
    
    def predict_vector(self, vector):
        pred_list = []
        for tree in self.forest:
            pred, prob = tree.predict_vector(vector)
            pred_list.append(pred)
        
        positive_prob = sum(pred_list) / len(pred_list)
        negative_prob = 1 - positive_prob
        if positive_prob >= negative_prob:
            return 1, positive_prob
        else:
            return 0, negative_prob
    
    def predict_matrix(self, matrix):
        pred_vector = []
        for vector in matrix:
            pred_vector.append(self.predict_vector(vector)[0])
        return np.array(pred_vector)

class cls_node():
    
    def __init__(self, feature_idx, best_guess, score, split_values, node_type, prob):
        self.feature_idx = feature_idx
        self.best_guess = best_guess
        self.score = score
        self.split_values = split_values
        self.node_type = node_type
        self.prob = prob
        self.children = dict()

    def print_node(self):
        print(f"feature: {self.feature_idx}" )
        print(f"best guess: {self.best_guess}")
        print(f"score: {self.score}")
        print(f"split values {self.split_values}")
        print(f"type: {self.node_type}")
        for child in self.children:
            if type(self.children[child]) == cls_node:
                print(f"child value: {self.split_values[child]}    child branch feature: {self.children[child].feature_idx}")
            else:
                print(f"child value: {self.split_values[child]}    child leaf: {self.children[child]}")
    
    def predict_vector(self, vector):
        current_node = self
        while type(current_node) == cls_node:
            current_idx = current_node.feature_idx
            node_type = current_node.node_type
            split_values = current_node.split_values
            current_value = vector[current_idx]
            
            if node_type == "discrete":
                if current_value not in split_values:
                    if 0 not in split_values:
                        return current_node.best_guess, current_node.prob
                    current_value = 0
                current_node = current_node.children[split_values.index(current_value)]
            
            else:
                if current_value < split_values:
                    current_node = current_node.children[0]
                else:
                    current_node = current_node.children[1]
        
        return current_node["best_guess"], current_node["prob"]

    def predict_matrix(self,matrix):
        pred_vector = []
        for vector in matrix:
            pred_vector.append(self.predict_vector(vector)[0])
        return np.array(pred_vector)

class custom_decision_tree():


    def __init__(self, features, labels, continious_size = 10, discrete_idx_list = [], min_samples = 2, max_depth = np.inf, method = "entropy", print_progress = True, max_features = None, weight = 0.5):
        self.feature_matrix = features
        self.label_vector = labels
        self.continious_size = continious_size
        self.discrete_idx_list = discrete_idx_list
        self.split_values = []
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.labaled_idx = 0
        self.progress = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        self.total_labels = len(labels)
        self.method = method
        self.print_progress = print_progress
        self.max_features = max_features
        self.weight = weight
        if self.max_features != None and self.max_features >= self.feature_matrix.shape[1]:
            self.max_features = None

    def calc_score(self,label_list):
        total = len(label_list)
        positive_prob = sum(label_list)/total
        negative_prob = 1 - positive_prob
        best_guess = int(positive_prob >= negative_prob) # if positive prob is equal or higher than negative prob guess 1
        
        if positive_prob == 0 or negative_prob == 0:
            score = 0
        elif self.method == "entropy":
            score = (-negative_prob * math.log2(negative_prob)) +(-positive_prob * math.log2(positive_prob))
        elif self.method == "gini":
            score = 1 - ((positive_prob**2) + (negative_prob**2))
        elif self.method == "prob":
            if best_guess:
                score =  negative_prob
            else:
                score = positive_prob
        return score, best_guess
    # https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8
    
    def discrete_gain(self, label_list, data_list, score):
        categories = Counter(data_list)
        weighted_score = 0
        
        for category in categories:
            temp_label_list = [label_list[i] for i, value in enumerate(data_list) if value == category]
            weighted_score += categories[category] / len(data_list) * self.calc_score(temp_label_list)[0]
        
        Info_gain = score - weighted_score
        split_values = list(set(data_list))
        
        return Info_gain, split_values

    def continious_gain(self, label_list, data_list, score, best_split_value = None):
        total = len(data_list)
        data_values = sorted(set(data_list))
        
        #some edge cases
        if len(data_values) == 1: #when only one value is present the information gain will always be 0
            return 0, None

        elif len(data_values) == 2:
            split_value = data_values[1]
            label_list_left = [label_list[i] for i in range(len(data_list)) if data_list[i] < split_value]
            label_list_right = [label_list[i] for i in range(len(data_list)) if data_list[i] >= split_value]
            weighted_score = len(label_list_left) / total * self.calc_score(label_list_left)[0] + len(label_list_right) / total * self.calc_score(label_list_right)[0]
            Info_gain = score - weighted_score
            return Info_gain, split_value

        elif max(data_values) == np.inf: #when a value is np.inf it doesn't occur in a sentence so it should create a seperate split
            split_value = np.inf
            label_list_left = [label_list[i] for i in range(len(data_list)) if data_list[i] < split_value]
            label_list_right = [label_list[i] for i in range(len(data_list)) if data_list[i] >= split_value]
            weighted_score = len(label_list_left) / total * self.calc_score(label_list_left)[0] + len(label_list_right) / total * self.calc_score(label_list_right)[0]
            Info_gain = score - weighted_score
            return Info_gain, split_value

        elif max(data_values) >= 0 and min(data_values) < 0: #negative a postive values should be split seperatly because grammer rules
            split_value = 0
            label_list_left = [label_list[i] for i in range(len(data_list)) if data_list[i] < split_value]
            label_list_right = [label_list[i] for i in range(len(data_list)) if data_list[i] >= split_value]
            weighted_score = len(label_list_left) / total * self.calc_score(label_list_left)[0] + len(label_list_right) / total * self.calc_score(label_list_right)[0]
            Info_gain = score - weighted_score
            return Info_gain, split_value

        
        min_idx = 1 # this forces at least a split in the data
        max_idx = len(data_values) - 1
        
        #start the split at the location of the previous best split as that is more likely to be close the correct split.
        if best_split_value == None:
            middle_idx = int(np.floor((min_idx+max_idx) / 2))
        else:
            middle_idx = data_values.index(best_split_value)
        

        # the best idx is most likely around the middle of the dataset
        # so we start by taking the middle value. Then check wheter going left or right will result in a higher entropy
        # then we only look at the values between the middle value and the left or right value. Repeat the previous steps on the new list of data
        # continue until only two idx are left
        while (max_idx - min_idx) > 1:
            
            #take the value inbetween the current middle idx and the min idx and calc the entropy of that split
            neg_middle = int(np.floor(min_idx + middle_idx) / 2)
            neg_middle_value = data_values[neg_middle]
            label_list_left = [label_list[i] for i in range(len(data_list)) if data_list[i] < neg_middle_value]
            label_list_right = [label_list[i] for i in range(len(data_list)) if data_list[i] >= neg_middle_value]
            neg_score = len(label_list_left) / total * self.calc_score(label_list_left)[0] + len(label_list_right) / total * self.calc_score(label_list_right)[0]
            
            #take the value inbetween the current middle idx and the max idx and calc the entropy of that split
            pos_middle = int(np.floor(max_idx + middle_idx) / 2)
            pos_middle_value = data_values[pos_middle]
            label_list_left = [label_list[i] for i in range(len(data_list)) if data_list[i] < pos_middle_value]
            label_list_right = [label_list[i] for i in range(len(data_list)) if data_list[i] >= pos_middle_value]
            pos_score = len(label_list_left) / total * self.calc_score(label_list_left)[0] + len(label_list_right) / total * self.calc_score(label_list_right)[0]

            #check which one has the lowest score
            if neg_score < pos_score:
                max_idx = middle_idx
            else:
                min_idx = middle_idx
            
            middle_idx = int(np.floor((min_idx+max_idx) / 2))
        
        # if min and max is the same value it will still return the correct info
        #look which one of the two left over splits have the lowest entropy
        min_value = data_values[min_idx]
        label_list_left = [label_list[i] for i in range(len(data_list)) if data_list[i] < min_value]
        label_list_right = [label_list[i] for i in range(len(data_list)) if data_list[i] >= min_value]
        min_score = len(label_list_left) / total * self.calc_score(label_list_left)[0] + len(label_list_right) / total * self.calc_score(label_list_right)[0]

        max_value = data_values[max_idx]
        label_list_left = [label_list[i] for i in range(len(data_list)) if data_list[i] < max_value]
        label_list_right = [label_list[i] for i in range(len(data_list)) if data_list[i] >= max_value]
        max_score = len(label_list_left)  /total * self.calc_score(label_list_left)[0] + len(label_list_right) / total * self.calc_score(label_list_right)[0] 
        
        if min_score < max_score:
            Info_gain = score - min_score
            split_values = min_value
        else:
            Info_gain = score - max_score
            split_values = max_value

        return Info_gain, split_values

    def find_best_split(self, idx_list, label_list, suggested_split = defaultdict(lambda:None)):
        best_ig = 0
        score = self.calc_score(label_list)[0]
        prev_best_split = defaultdict(lambda:None)
        feature_idx_list = [idx for idx in range(self.feature_matrix.shape[1])]
        if self.max_features == None:
            random_features = feature_idx_list
        else:
            feature_count = np.random.randint(1, self.max_features)
            random_features = np.random.choice(feature_idx_list, feature_count, replace=False)

        for feature_idx in random_features:
            
            data_list = [self.feature_matrix[idx,feature_idx] for idx in idx_list]
            
            if feature_idx in self.discrete_idx_list:
                Info_gain = self.discrete_gain(label_list, data_list, score)
                node_type = "discrete"
            else:
                Info_gain = self.continious_gain(label_list, data_list, score, suggested_split[feature_idx])
                node_type = "continious"
                if Info_gain[0] != np.inf:
                    prev_best_split[feature_idx] = Info_gain[0]
            
            if Info_gain[0] > best_ig:
                best_ig = Info_gain[0]
                best_values = Info_gain[1]
                best_idx = feature_idx
                best_type = node_type
        
        #if there is now information gain a leaf node will be created
        if best_ig == 0:
            best_type = "leaf"
            best_idx = None
            best_values = None    
            
        return best_idx, best_type, best_values, suggested_split
    

    def build_node(self, idx_list, node, sugested_split, depth):
        feature_idx = node.feature_idx
        i = 0
        while True:

            #find which data points quallify for this split
            if feature_idx in self.discrete_idx_list:
                if len(node.split_values) == i:
                    break
                split_value = node.split_values[i]
                temp_idx_list = [idx for idx in idx_list if self.feature_matrix[idx,feature_idx] == split_value]
            else:
                split_value = node.split_values
                if i == 2:
                    break
                elif i == 0:
                    temp_idx_list = [idx for idx in idx_list if self.feature_matrix[idx,feature_idx] < split_value]
                else:
                    temp_idx_list = temp_idx_list = [idx for idx in idx_list if self.feature_matrix[idx,feature_idx] >= split_value]
            
            #get the labels of the current split
            label_list = [self.label_vector[idx] for idx in temp_idx_list]
            
            #get the probability and best guess of current split
            positve_prob = sum(label_list) / len(label_list)
            negative_prob = 1 - positve_prob
            score, best_guess = self.calc_score(label_list)
            if best_guess:
                prob = positve_prob
            else:
                prob = negative_prob
           
           #when there are less samples than minimum required create a leaf node
            if len(temp_idx_list) < self.min_samples:    
                node.children[i] = {"best_guess" : best_guess, "prob": prob}
                self.labaled_idx += len(temp_idx_list)
            #when tree reaches max depth create a leaf node
            elif depth > self.max_depth:
                node.children[i] = {"best_guess" : best_guess, "prob": prob}
                self.labaled_idx += len(temp_idx_list)
            #if the score is 0 that means that the split only contains 1 label so you can create a leaf node
            elif score == 0:
                node.children[i] = {"best_guess": best_guess, "prob": prob}
                self.labaled_idx += len(temp_idx_list)
            #if the above three cases don't hold find the best split
            else:
                temp_idx, node_type, split_values, new_sugested_split = self.find_best_split(temp_idx_list, label_list, sugested_split)
                if node_type == "leaf":
                    node.children[i] = {"best_guess": best_guess, "prob": prob}
                    self.labaled_idx += len(temp_idx_list)
                    i += 1
                    continue
                
                #after finding the best split build new node using current split
                temp_node = cls_node(temp_idx, best_guess, score, split_values, node_type, prob)
                temp_node = self.build_node(temp_idx_list.copy(),temp_node, new_sugested_split, depth = depth + 1)
                node.children[i] = temp_node
            
            #print statement to check progress
            if len(self.progress) != 0 and self.labaled_idx/self.total_labels > self.progress[0] and self.print_progress:
                print(self.progress[0])
                self.progress.pop(0)
            
            i += 1

        return node
    
    def build_tree(self):
        idx_list = [i for i in range(len(self.feature_matrix))]
        label_list = [self.label_vector[idx] for idx in idx_list]
        prob = sum(label_list)/len(label_list)
        best_idx, node_type, split_values, sugested_split = self.find_best_split(idx_list, label_list)
        entropy, best_guess = self.calc_score(label_list)
        #build the tree recursively. 
        self.tree = cls_node(best_idx, best_guess, entropy, split_values, node_type, prob)
        self.tree = self.build_node(idx_list.copy(), self.tree, sugested_split, 1)

class random_forest():
    def __init__(self, features, labels, continious_size = 10, discrete_idx_list = [], min_samples = 2, method = "entropy", print_progress = True):
        self.feature_matrix = features
        self.label_vector = labels
        self.continious_size = continious_size
        self.discrete_idx_list = discrete_idx_list
        self.min_samples = min_samples
        self.method = method
        self.print_progress = print_progress

    def get_random_points(self):
        data_points = self.feature_matrix.shape[0]
        idx_list = np.random.choice(data_points,data_points)
        self.random_feature_matrix = []
        self.random_label_vector = []
        for idx in idx_list:
            self.random_feature_matrix.append(self.feature_matrix[idx])
            self.random_label_vector.append(self.label_vector[idx])
    

    def build_forest(self, tree_count = 10):
        self.forest = cls_forest()
        for i in range(tree_count):
            if self.print_progress:
                print(i)
            dt = custom_decision_tree(self.feature_matrix, self.label_vector, self.continious_size, self.discrete_idx_list, self.min_samples, self.method, self.print_progress)
            dt.build_tree()
            self.forest.forest.append(dt.tree)





# C:\Users\karst\OneDrive\Documenten\uni\jaar 3 semester 2, het semester dat ik opgaf\Bachelor-graduation-project\code\notebooks\feature_matrix.npy
#  
# feature_matrix = np.load(map_path + "feature_matrix.npy")
# label_matrix = np.load(map_path + "label_matrix.npy")
# sentence_idx = list(np.load(map_path + "sentence_idx.npy"))

# total_sentences = len(set(sentence_idx))
# split = 0.5
# rng = default_rng()
# training_sentence_ids = rng.choice(total_sentences, size=int(total_sentences*split), replace=False)
# test_sentence_ids = [i for i in range(total_sentences) if i not in training_sentence_ids]

# training_id_list = []
# test_id_list = []
# for i, item in enumerate(sentence_idx):
#     if item in training_sentence_ids:
#         training_id_list.append(i)
#     else:
#         test_id_list.append(i)

# training_feature_matrix = np.array([feature_matrix[i] for i in training_id_list])
# training_label_matrix = np.array([label_matrix[i] for i in training_id_list])
# test_feature_matrix = np.array([feature_matrix[i] for i in test_id_list])
# test_label_matrix = np.array([label_matrix[i] for i in test_id_list])


# test_array = np.array([
#     [0,-3,5,2,1], # 0
#     [0,-2,6,2,0], # 0
#     [0,-1,7,2,1], # 1
#     [0,0,4,1,0], # 1
#     [1,1,3,1,1], # 0
#     [1,2,2,1,0], # 0
#     [1,3,8,0,1], # 1
#     [1,4,1,0,0], # 1
# ])
# test_array_labels = np.array([0,0,1,1,0,0,1,1])
# total_features = training_feature_matrix.shape[1]
# discrete_idx_list = [total_features-2, total_features-3, total_features-4, total_features-5, total_features-6]
# test = custom_decision_tree(training_feature_matrix, training_label_matrix, discrete_idx_list=discrete_idx_list, min_samples=35)
# # test = custom_decision_tree(test_array,test_array_labels,discrete_idx_list=[0,3,4])
# test.build_tree()
# tree = test.tree
# # with open(map_path + 'custom_tree_3.pkl', 'wb') as outp:
# #     pickle.dump(tree, outp)

# tree.print_node()
# prediction = tree.predict_matrix(test_feature_matrix)
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# for i , pred in enumerate(prediction):
#     true_label = test_label_matrix[i]
#     if pred:
#         if pred == true_label:
#             tp += 1
#         else:
#             fp += 1
#     else:
#         if pred == true_label:
#             tn += 1
#         else:
#             fn += 1
# precision = tp / (tp+fp)
# recall = tp / (tp+fn)
# F_1 = 2*tp / (2*tp + fp + fn)
# print(precision)
# print(recall)
# print(F_1)


# print(tree.predict_vector())

# print(len(test_array.shape))
# print(len(test_array_labels.shape))

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