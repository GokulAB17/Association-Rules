#Prepare rules for the all the data sets 
#1) Try different values of support and confidence. 
#Observe the change in number of rules for different support,confidence values
#2) Change the minimum length in apriori algorithm
#3) Visulize the obtained rules using different plots 
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

#load movies.csv dataset
movies=pd.read_csv(r"filepath\my_movies.csv")

#EDA
movies.head()
movies.isnull().sum()

#Data Preprocessing
movies1=pd.DataFrame(movies.iloc[:,5:15])

#Applying apriori algorithm
frequent_itemsets = apriori(movies1, min_support=0.005, max_len=3,use_colnames = True)


#Plotiing first 10 itemsets by support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(35,8));plt.bar(x = list(range(1,10)),height = frequent_itemsets.support[1:10],color='rgmyk');plt.xticks(list(range(1,10)),frequent_itemsets.itemsets[1:10])
plt.xlabel('item-sets');plt.ylabel('support')

#Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#no of rules 124
rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
#no of rules 110

#################Using rules based on lift#################
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

######################Redundancy removing ###############################
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
finalrules=rules_no_redudancy.sort_values('lift',ascending=False)
