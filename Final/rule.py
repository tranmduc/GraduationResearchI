# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 03:59:03 2020

@author: Minh Duc
"""

class Rule:

    def __init__(self, rule, target=None):
        """
        :param rule: Tuple with 0s & 1s, representing the rule. (ex. (0,0,1,0,1,1,0,1))
        :param target: Target of rule. (ex. "Iris-Setosa")
        :return:
        """
        self.rule = rule
        self.target = target

    def __eq__(self, other):
        #return self.__hash__() == other.__hash__()
    
        return self.rule==other.rule and self.target==other.target

    def __hash__(self):
        #val = bytearray(self.target.encode() if self.target is not None else "".encode())+bytearray(self.rule)
        #return int.from_bytes(val, byteorder='big')
    
        return hash(('rule', self.rule[0] + self.rule[1] + self.rule[2] + self.rule[3],'target', self.target))

    def __str__(self):
        ret_val = 'If '
        for value in self.rule:
            ret_val += str(value)
            ret_val += " and "

        return ret_val[:-4] + "then target is " + str(self.target)