#Prepare rules for the all the data sets 
#1) Try different values of support and confidence. 
#Observe the change in number of rules for different support,confidence values
#2) Change the minimum length in apriori algorithm
#3) Visulize the obtained rules using different plots 
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules


groceries = []
# As the file is in transaction data we will be reading data directly 
with open(r"filepath\groceries.csv") as f:
    groceries = f.read()


#DATA PREPROCESSING
# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
    

all_groceries_list = [i for item in groceries_list for i in item]
from collections import Counter

item_frequencies = Counter(all_groceries_list)
# after sorting
#item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5));plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("items")
plt.ylabel("Count")


# Creating Data Frame for the transactions data 

# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')


####################applying apriori algorithm to preprocessed data set################
frequent_itemsets = apriori(X, min_support=0.009, max_len=3,use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.figure(figsize=(15,5));plt.bar(x = list(range(1,5)),height = frequent_itemsets.support[1:5],color='rgmyk');plt.xticks(list(range(1,5)),frequent_itemsets.itemsets[1:5])
plt.xlabel('item-sets');plt.ylabel('support')

# Most Frequent item sets based on confidence
frequent_itemsets.sort_values('confidence',ascending = False,inplace=True)
plt.figure(figsize=(15,5));plt.bar(x = list(range(1,5)),height = frequent_itemsets.support[1:5],color='rgmyk');plt.xticks(list(range(1,5)),frequent_itemsets.itemsets[1:5])
plt.xlabel('item-sets');plt.ylabel('confidence')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#no of rules 796
rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
#no of rules created 588
rules2 = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
#no of rules created 618
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

 
########################## To eliminate Redudancy in Rules #################################### 
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
finalrules=rules_no_redudancy.sort_values('lift',ascending=False).head(10)
