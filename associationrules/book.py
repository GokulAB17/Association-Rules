#Prepare rules for the all the data sets 
#1) Try different values of support and confidence. 
#Observe the change in number of rules for different support,confidence values
#2) Change the minimum length in apriori algorithm
#3) Visulize the obtained rules using different plots 

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

#loading dataset of book.csv
books=pd.read_csv(r"filepath\book.csv")
books.head()

#chceking null entries
books.isnull().sum()

#As dataset is in proper format to apply apriori algoritmn so data preprocessing not required

frequent_itemsets = apriori(books, min_support=0.005, max_len=3,use_colnames = True)

frequent_itemsets.sort_values('support',ascending = False,inplace=True)

# barplot of top 10 
import matplotlib.pyplot as plt

plt.figure(figsize=(35,8));plt.bar(x = list(range(1,10)),height = frequent_itemsets.support[1:10],color='rgmyk');plt.xticks(list(range(1,10)),frequent_itemsets.itemsets[1:10])
plt.xlabel('item-sets');plt.ylabel('support')

#we can create rules by confidence or support or lift metrics and setting threshold for the same 
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
# no of rules created 1054
rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
#no of rules created 556
rules2 = association_rules(frequent_itemsets, metric="support", min_threshold=0.005)
#no of rules created 1058
rules.head(20)

#Using lift rules for further processing
#sort rules according to lift in descending order
rules.sort_values('lift',ascending = False,inplace=True)

 
########################## To eliminate Redudancy in Rules ############################
#function to sort and covert to list
def to_list(i):
    return (sorted(list(i)))

#concatenate antecedents and consequents in one series and sort it
ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)

#sort each row in Series ma_X
ma_X = ma_X.apply(sorted)

#making it list
rules_sets = list(ma_X)

#finding unique rules in rules_set and getting index no also 
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
finalrules=rules_no_redudancy.sort_values('lift',ascending=False).head(10)

#Sorting them with confidence and getting top 10 rules
finalrules1=rules_no_redudancy.sort_values('confidence',ascending=False).head(10)


#Sorting them with support and getting top 10 rules
finalrules2=rules_no_redudancy.sort_values('support',ascending=False).head(10)
