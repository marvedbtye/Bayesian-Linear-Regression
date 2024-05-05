# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 19:07:40 2024

@author: Carlos Mondejar
"""

# calculate the probability of cancer patient diagnostic test

# calculate P(A|B) given P(A), P(B|A), P(B|not A)
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
 # calculate P(not A)
 not_a = 1 - p_a
 # calculate P(B)
 p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
 #calculate P(A|B)
 p_a_given_b = (p_b_given_a * p_a) / p_b
 return p_a_given_b

# P(A)
p_a = 0.0002
# P(B|A)
p_b_given_a = 0.85
# P(B|not A)
p_b_given_not_a = 0.05
#calculate P(A|B)
result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)
#summarize
print('P(A|B) = %.3f%%' % (result * 100))