import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from IPython.display import display_html

def toy_dataset():
    data = [['Milk','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
             ['Dill','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
             ['Milk','Apple','Kidney Beans','Eggs'],
             ['Milk','Unicorn','Corn','Kidney Beans','Yogurt'],
             ['Corn','Onion','Onion','Kidney Beans','Ice Cream','Eggs']
    ]
    print('do you want to view the raw data?')
    choice = input()
    if choice == 'yes':
        print('Raw Data')
        print(data)
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary,columns=te.columns_)
    print('Do you want to view the Encoded data:?')
    choice = input()
    if choice == 'yes':
        print('Encoded Data:')
        display_html(df)
    return df

def frequent_itemset(data):
    print('Enter the value of minimum support threshold:')
    support = float(input())
    frequent_itemsets = fpgrowth(data,min_support=support,use_colnames=True)
    print("Do you want to view freequent itemsets geneerated by FP-growth?")
    choice = input()
    if choice == 'yes':
        print('Frequent itemset:')
        display_html(frequent_itemsets)
    return frequent_itemsets
print('*'*20)

def association_rule(frequent_itemsets):
    print('Enter your metric of interest confidence or lift:')
    choice = input()
    if choice == 'confidence':
        print('Enter minimum confidence threshold value:')
        min_confidence = float(input())
        rule = association_rules(frequent_itemsets,metric ='confidence',min_threshold = min_confidence)
    elif choice == 'lift':
        print('Enter minimum lift threshold value:')
        min_lift = float(input())
        rule = association_rule(frequent_itemsets,metric='lift',min_threshold=min_lift)
    print('Do you want to view the learned association rules?')
    choice = input()
    if choice == 'yes':
        display_html(rule.drop(['leverage','conviction'],axis=1))
    else:
        quit()

def main():
    data = toy_dataset()
    frequent_itemsets = frequent_itemset(data)
    association_rule(frequent_itemsets)
main()

        