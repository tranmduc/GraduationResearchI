# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 03:50:45 2020

@author: Minh Duc
"""

import numpy as np
import copy
from rule import Rule
import general_functions as gf
import pandas as pd

class Rbfc:

    def __init__(self, data_set, targets, col_num, fuzzy_set_size, minmax):
        self.data_set = data_set
        self.targets = targets
        
        self.col_num = col_num

        if fuzzy_set_size == 3:
            self.set_fuzzy = ["small", "medium", "large"]
        if fuzzy_set_size == 5:
            self.set_fuzzy = ["very small", "small", "medium", "large", "very large"]
        if fuzzy_set_size == 7:
            self.set_fuzzy = ["very very small", "very small", "small", "medium", "large", "very large", "very very large"]
            
        self.set = len(self.set_fuzzy)
        self.xmin = []
        self.xmax = []
        for i in range (self.col_num):
            self.xmin.append(minmax[i][0])
            self.xmax.append(minmax[i][1])
        
        self.step = [(self.xmax[i]-self.xmin[i])/(self.set-1) for i in range(len(self.xmax))]

        self._sort_data_set()
        self.membership()
        self.gen_rules()

    def _sort_data_set(self):
        """
        Sorts dataset's individual columns and inserts them in self._sorted_data_set.
        """
        self._sorted_data_set = copy.copy(self.data_set)
        for i in range(self._sorted_data_set.shape[1]):
            self._sorted_data_set[self._sorted_data_set[:, i].sort()]

    def print_rules(self):
        i=0
        for rule in self.not_dup_rules:
            print(rule)
            i = i+1
        print(i)
        
    def show_rules(self):
        return self.not_dup_rules
      
    def membership(self):
        self.ux = []
        for i in range(self.col_num):
            self.ux.append(pd.DataFrame(np.zeros([len(self.data_set),self.set])))
        
        for i in range(self.col_num):
            for s in range(self.set):
                av=np.arange(self.xmin[i],self.xmax[i]+self.step[i],self.step[i])
                self.ux[i].loc[:][s]=gf.tri_function(self.data_set[:,i],av[s],self.step[i])
            
    def gen_rules(self):
        self.WM_rule=pd.DataFrame(np.zeros([len(self.data_set),self.col_num]))
        for i in range(len(self.data_set)):
            for j in range(self.col_num):
                self.WM_rule.loc[i][j]=self.ux[j].idxmax(axis=1)[i]#x fuzzy

        self.rules = []
        self.not_dup_rules = []
        for i in range(len(self.data_set)):
            #fuzzy_label = [" "," "," "," "]

            fuzzy_label = [" "]*self.col_num
            for j in range(self.col_num):
                for k in range(self.set):
                    if self.WM_rule.loc[i][j] == k:
                        fuzzy_label[j] = self.set_fuzzy[k]
                    
            rule_gen = Rule(rule=fuzzy_label, target=self.targets[i])
            
            self.rules.append(rule_gen)
            
            self.not_dup_rules = gf.remove_doubles(self.rules)
            
        self.target_categories = gf.remove_doubles(self.targets)
        
            
    def gen_rule(self, sample, target=None):
        #fuzzy_label = ["","","",""]
        fuzzy_label = [" "]*self.col_num
        for i in range(self.col_num):
            for j in range(self.set):
                    if sample[i] == j:
                        fuzzy_label[i] = self.set_fuzzy[j]
                
        return Rule(rule=fuzzy_label, target=target)
    
    def predict(self, samples):
        # Used to hold classifications of all samples for return
        classifications = []
        # Used to hold category name as key and number of times category
        # was found during classification of sample (ex. categories{"setosa": 2, "virginia": 2, ...}
        categories = {}
        
        uxX = []
        for i in range(self.col_num):
            uxX.append(pd.DataFrame(np.zeros([len(samples),self.set])))

        for i in range(self.col_num):
            for s in range(self.set):
                av=np.arange(self.xmin[i],self.xmax[i]+self.step[i],self.step[i])
                uxX[i].loc[:][s]=gf.tri_function(samples[:,i],av[s],self.step[i])

        WM_rule_new=pd.DataFrame(np.zeros([len(samples),self.col_num]))
        
        for i in range(len(samples)):
            for j in range(self.col_num):
                WM_rule_new.loc[i][j]=uxX[j].idxmax(axis=1)[i]#x fuzzy

        for i in range(len(samples)):
            # Initialize categories values to 0
            for cat in self.target_categories:
                categories[cat] = 0
                    
            fuzzy_label = [" "]*self.col_num
            for j in range(self.col_num):
                for k in range(self.set):
                    if WM_rule_new.loc[i][j] == k:
                        fuzzy_label[j] = self.set_fuzzy[k]
                    
            sample_rule = Rule(rule=fuzzy_label)

            # Classify sample
            for rule in self.rules:
                if (sample_rule.rule==rule.rule):
                    categories[rule.target] += 1
        

            # Sort categories dict base on its values
            classifications.append(sorted(categories, key=categories.get, reverse=True)[0])
        # print("classify samples",classifications)
        return np.array(classifications)       