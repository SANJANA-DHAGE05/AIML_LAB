#TERMWORK-4  (FIND S)

import csv

num_attributes = 4
a = []
print("\n The Given Data Set \n")

with open('data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a.append (row)
        print(row)
        
print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)

for j in range(0,num_attributes):
    hypothesis[j] = a[0][j];

print("\n Find S: Finding a Maximally Specific Hypothesis\n")

for i in range(0,len(a)):
    if a[i][num_attributes]=='yes':
        for j in range(0,num_attributes):
             if a[i][j]!=hypothesis[j]:
                 hypothesis[j]='?'
             else:
                hypothesis[j]= a[i][j]
    print(" For instance No:{0} the hypothesis is".format(i),hypothesis)

print("\n The Maximally Specific Hypothesis for a given Examples :\n")
print(hypothesis)

'''
---------OUPUT----------

The Given Data Set 

['green', 'hard', 'no', 'wrinkled', 'yes']
['green', 'hard', 'yes', 'smooth', 'no']
['brown', 'soft', 'no', 'wrinkled', 'no']
['orange', 'hard', 'no', 'wrinkled', 'yes']
['green', 'soft', 'yes', 'smooth', 'yes']
['green', 'hard', 'yes', 'wrinkled', 'yes']
['orange', 'hard', 'no', 'wrinkled', 'yes']

 The initial value of hypothesis: 
['0', '0', '0', '0']

 Find S: Finding a Maximally Specific Hypothesis

 For instance No:0 the hypothesis is ['green', 'hard', 'no', 'wrinkled']
 For instance No:1 the hypothesis is ['green', 'hard', 'no', 'wrinkled']
 For instance No:2 the hypothesis is ['green', 'hard', 'no', 'wrinkled']
 For instance No:3 the hypothesis is ['?', 'hard', 'no', 'wrinkled']
 For instance No:4 the hypothesis is ['?', '?', '?', '?']
 For instance No:5 the hypothesis is ['?', '?', '?', '?']
 For instance No:6 the hypothesis is ['?', '?', '?', '?']

 The Maximally Specific Hypothesis for a given Examples :

['?', '?', '?', '?']

'''
