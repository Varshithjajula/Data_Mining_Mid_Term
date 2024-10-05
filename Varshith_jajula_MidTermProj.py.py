#!/usr/bin/env python
# coding: utf-8

# # Data Mining: Comparative Analysis of Algorithms

# This notebook is designed to guide you through a comparative analysis of three different data mining algorithms: **Brute Force**, **Apriori**, and **FP-Growth**.
# 
# In the field of data mining, these algorithms are commonly used for frequent itemset mining and association rule generation. Frequent itemset mining identifies sets of items that appear frequently together in transaction datasets, while association rule generation discovers interesting relationships between variables in large databases.
# 
# Throughout this notebook, we will:
# 
# 1. **Create Transactional Databases**: Generate datasets containing items typically found in various stores, including Amazon, Walmart, Best Buy, Nike,K-Mart, and Domino's.
# 2. **Implement Brute Force Algorithm**: Use a straightforward method to find frequent itemsets and generate association rules from the transaction data.
# 3. **Implement Apriori Algorithm**: Apply the Apriori technique to identify frequent itemsets and derive association rules based on user-defined support and confidence thresholds.
# 4. **Implement FP-Growth Algorithm**: Utilize the FP-Growth method for efficient frequent itemset mining and association rule generation.
# 5. **User Input**: Allow users to select different transactional databases for analysis, also the  minimum confidence threshold and minimum support threshold.
# 6. **Compare Results and Performance**: Analyze and compare the outcomes and efficiency of the three algorithms to understand their strengths and weaknesses in different scenarios.
# 
# By the end of this notebook, you will gain insights into the practical application of these algorithms and their effectiveness in mining associations from transactional data.
# 

# ## Part-1 : Creation Of Databases

# In[1]:


# Imports
import csv
import itertools
import time 
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


store_items = {'Amazon': ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'Android Programming: The Big Nerd Ranch', 'Head First Java 2nd Edition', 'Beginning Programming with Java', 'Java 8 Pocket Guide', 'C++ Programming in Easy Steps', 'Effective Java (2nd Edition)', 'HTML and CSS: Design and Build Websites'],
'Best Buy': ['Digital Camera', 'Lab Top', 'Desk Top', 'Printer', 'Flash Drive', 'Microsoft Office', 'Speakers', 'Lab Top Case', 'Anti-Virus', 'External Hard-Drive'],
'K-mart': ['Quilts', 'Bedspreads', 'Decorative Pillows', 'Bed Skirts', 'Sheets', 'Shams', 'Bedding Collections', 'Kids Bedding', 'Embroidered Bedspread', 'Towels'],
'Nike': ['Running Shoe', 'Soccer Shoe', 'Socks', 'Swimming Shirt', 'Dry Fit V-Nick', 'Rash Guard', 'Sweatshirts', 'Hoodies', 'Tech Pants', 'Modern Pants'],
'Dominos': ['Pizza', 'Garlic Bread', 'Coke', 'Pepsi', 'Chicken Wings', 'Pasta', 'Cheese Sticks', 'Salad', 'Brownie', 'Dessert Pizza']}

predefined_transactions = {
    'Amazon': [['trans1', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'Android Programming: The Big Nerd Ranch']], ['trans2', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies']], ['trans3', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'Android Programming: The Big Nerd Ranch', 'Head First Java 2nd Edition']], ['trans4', ['Android Programming: The Big Nerd Ranch', 'Head First Java 2nd Edition', 'Beginning Programming with Java']], ['trans5', ['Android Programming: The Big Nerd Ranch', 'Beginning Programming with Java', 'Java 8 Pocket Guide']], ['trans6', ['A Beginner’s Guide', 'Android Programming: The Big Nerd Ranch', 'Head First Java 2nd Edition']], ['trans7', ['A Beginner’s Guide', 'Head First Java 2nd Edition', 'Beginning Programming with Java']], ['trans8', ['Java: The Complete Reference', 'Java For Dummies', 'Android Programming: The Big Nerd Ranch']], ['trans9', ['Java For Dummies', 'Android Programming: The Big Nerd Ranch', 'Head First Java 2nd Edition', 'Beginning Programming with Java']], ['trans10', ['Beginning Programming with Java', 'Java 8 Pocket Guide', 'C++ Programming in Easy Steps']], ['trans11', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'Android Programming: The Big Nerd Ranch']], ['trans12', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'HTML and CSS: Design and Build Websites']], ['trans13', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'Java 8 Pocket Guide', 'HTML and CSS: Design and Build Websites']], ['trans14', ['Java For Dummies', 'Android Programming: The Big Nerd Ranch', 'Head First Java 2nd Edition']], ['trans15', ['Java For Dummies', 'Android Programming: The Big Nerd Ranch']], ['trans16', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'Android Programming: The Big Nerd Ranch']], ['trans17', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies', 'Android Programming: The Big Nerd Ranch']], ['trans18', ['Head First Java 2nd Edition', 'Beginning Programming with Java', 'Java 8 Pocket Guide']], ['trans19', ['Android Programming: The Big Nerd Ranch', 'Head First Java 2nd Edition']], ['trans20', ['A Beginner’s Guide', 'Java: The Complete Reference', 'Java For Dummies']]],
    'Best Buy': [['trans1', ['Desk Top', 'Printer', 'Flash Drive', 'Microsoft Office', 'Speakers', 'Anti-Virus']], ['trans2', ['Lab Top', 'Flash Drive', 'Microsoft Office', 'Lab Top Case', 'Anti-Virus']], ['trans3', ['Lab Top', 'Printer', 'Flash Drive', 'Microsoft Office', 'Anti-Virus', 'Lab Top Case', 'External Hard-Drive']], ['trans4', ['Lab Top', 'Printer', 'Flash Drive', 'Anti-Virus', 'External Hard-Drive', 'Lab Top Case']], ['trans5', ['Lab Top', 'Flash Drive', 'Lab Top Case', 'Anti-Virus']], ['trans6', ['Lab Top', 'Printer', 'Flash Drive', 'Microsoft Office']], ['trans7', ['Desk Top', 'Printer', 'Flash Drive', 'Microsoft Office']], ['trans8', ['Lab Top', 'External Hard-Drive', 'Anti-Virus']], ['trans9', ['Desk Top', 'Printer', 'Flash Drive', 'Microsoft Office', 'Lab Top Case', 'Anti-Virus', 'Speakers', 'External Hard-Drive']], ['trans10', ['Digital Camera', 'Lab Top', 'Desk Top', 'Printer', 'Flash Drive', 'Microsoft Office', 'Lab Top Case', 'Anti-Virus', 'External Hard-Drive', 'Speakers']], ['trans11', ['Lab Top', 'Desk Top', 'Lab Top Case', 'External Hard-Drive', 'Speakers', 'Anti-Virus']], ['trans12', ['Digital Camera', 'Lab Top', 'Lab Top Case', 'External Hard-Drive', 'Anti-Virus', 'Speakers']], ['trans13', ['Digital Camera', 'Speakers']], ['trans14', ['Digital Camera', 'Desk Top', 'Printer', 'Flash Drive', 'Microsoft Office']], ['trans15', ['Printer', 'Flash Drive', 'Microsoft Office', 'Anti-Virus', 'Lab Top Case', 'Speakers', 'External Hard-Drive']], ['trans16', ['Digital Camera', 'Flash Drive', 'Microsoft Office', 'Anti-Virus', 'Lab Top Case', 'External Hard-Drive', 'Speakers']], ['trans17', ['Digital Camera', 'Lab Top', 'Lab Top Case']], ['trans18', ['Digital Camera', 'Lab Top Case', 'Speakers']], ['trans19', ['Digital Camera', 'Lab Top', 'Printer', 'Flash Drive', 'Microsoft Office', 'Speakers', 'Lab Top Case', 'Anti-Virus']], ['trans20', ['Digital Camera', 'Lab Top', 'Speakers', 'Anti-Virus', 'Lab Top Case']]],
    'K-mart': [['trans1', ['Decorative Pillows', 'Quilts', 'Embroidered Bedspread']], ['trans2', ['Embroidered Bedspread', 'Shams', 'Kids Bedding', 'Bedding Collections', 'Bed Skirts', 'Bedspreads', 'Sheets']], ['trans3', ['Decorative Pillows', 'Quilts', 'Embroidered Bedspread', 'Shams', 'Kids Bedding', 'Bedding Collections']], ['trans4', ['Kids Bedding', 'Bedding Collections', 'Sheets', 'Bedspreads', 'Bed Skirts']], ['trans5', ['Decorative Pillows', 'Kids Bedding', 'Bedding Collections', 'Sheets', 'Bed Skirts', 'Bedspreads']], ['trans6', ['Bedding Collections', 'Bedspreads', 'Bed Skirts', 'Sheets', 'Shams', 'Kids Bedding']], ['trans7', ['Decorative Pillows', 'Quilts']], ['trans8', ['Decorative Pillows', 'Quilts', 'Embroidered Bedspread']], ['trans9', ['Bedspreads', 'Bed Skirts', 'Shams', 'Kids Bedding', 'Sheets']], ['trans10', ['Quilts', 'Embroidered Bedspread', 'Bedding Collections']], ['trans11', ['Bedding Collections', 'Bedspreads', 'Bed Skirts', 'Kids Bedding', 'Shams', 'Sheets']], ['trans12', ['Decorative Pillows', 'Quilts']], ['trans13', ['Embroidered Bedspread', 'Shams']], ['trans14', ['Sheets', 'Shams', 'Bed Skirts', 'Kids Bedding']], ['trans15', ['Decorative Pillows', 'Quilts']], ['trans16', ['Decorative Pillows', 'Kids Bedding', 'Bed Skirts', 'Shams']], ['trans17', ['Decorative Pillows', 'Shams', 'Bed Skirts']], ['trans18', ['Quilts', 'Sheets', 'Kids Bedding']], ['trans19', ['Shams', 'Bed Skirts', 'Kids Bedding', 'Sheets']], ['trans20', ['Decorative Pillows', 'Bedspreads', 'Shams', 'Sheets', 'Bed Skirts', 'Kids Bedding']]],
    'Nike': [['trans1', ['Running Shoe', 'Socks', 'Sweatshirts', 'Modern Pants']], ['trans2', ['Running Shoe', 'Socks', 'Sweatshirts']], ['trans3', ['Running Shoe', 'Socks', 'Sweatshirts', 'Modern Pants']], ['trans4', ['Running Shoe', 'Socks', 'Sweatshirts', 'Tech Pants']], ['trans5', ['Dry Fit V-Nick', 'Swimming Shirt', 'Rash Guard', 'Sweatshirts']], ['trans6', ['Swimming Shirt', 'Hoodies', 'Dry Fit V-Nick', 'Tech Pants']], ['trans7', ['Swimming Shirt', 'Dry Fit V-Nick', 'Tech Pants']], ['trans8', ['Running Shoe', 'Socks', 'Sweatshirts', 'Hoodies']], ['trans9', ['Hoodies', 'Dry Fit V-Nick', 'Sweatshirts']], ['trans10', ['Swimming Shirt', 'Dry Fit V-Nick', 'Sweatshirts', 'Hoodies']], ['trans11', ['Sweatshirts', 'Modern Pants']], ['trans12', ['Sweatshirts', 'Hoodies']], ['trans13', ['Soccer Shoe', 'Sweatshirts']], ['trans14', ['Running Shoe', 'Soccer Shoe', 'Sweatshirts']], ['trans15', ['Running Shoe', 'Hoodies']], ['trans16', ['Running Shoe', 'Hoodies', 'Dry Fit V-Nick']], ['trans17', ['Running Shoe', 'Sweatshirts']], ['trans18', ['Sweatshirts', 'Hoodies', 'Tech Pants']], ['trans19', ['Hoodies', 'Modern Pants']], ['trans20', ['Sweatshirts', 'Modern Pants', 'Tech Pants']]],
    'Dominos': [['trans1', ['Pizza', 'Coke']], ['trans2', ['Garlic Bread', 'Pepsi', 'Pizza']], ['trans3', ['Pizza', 'Garlic Bread', 'Pepsi']], ['trans4', ['Pizza', 'Pepsi']], ['trans5', ['Pizza', 'Garlic Bread', 'Coke']], ['trans6', ['Pizza', 'Coke', 'Garlic Bread']], ['trans7', ['Coke', 'Pizza', 'Pepsi', 'Garlic Bread']], ['trans8', ['Coke', 'Pizza', 'Garlic Bread']], ['trans9', ['Coke', 'Garlic Bread']], ['trans10', ['Pizza', 'Garlic Bread']], ['trans11', ['Pizza', 'Coke']], ['trans12', ['Pizza', 'Pepsi']], ['trans13', ['Pizza', 'Coke']], ['trans14', ['Garlic Bread', 'Pepsi']], ['trans15', ['Garlic Bread', 'Coke', 'Pizza']], ['trans16', ['Garlic Bread', 'Pepsi']], ['trans17', ['Pizza', 'Pepsi', 'Coke']], ['trans18', ['Pizza', 'Coke', 'Pepsi']], ['trans19', ['Pizza']], ['trans20', ['Pizza', 'Coke', 'Pepsi']]]
}

def save_transactions_to_csv(transactions):
    for store, trans_list in transactions.items():
        with open(f'{store}_transactions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(trans_list)
        print(f"Generated CSV for {store}: {store}_transactions.csv")

def main(predefined_transactions):
    save_transactions_to_csv(predefined_transactions)

main(predefined_transactions)


# ## Part-2: Implementing the Brute Force Method for Frequent Itemset Mining and Association Rule Generation & Part 3: Implementation of Apriori and FP-Growth methods. 
# 

# In[3]:


def read_transactions(filename):
    transactions = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) > 1:
                transactions.append(row[1].strip("[]").replace("'", "").split(", "))
    return transactions

def count_frequency(itemset, transactions):
    return sum(1 for transaction in transactions if set(itemset).issubset(transaction))

def generate_itemsets(items, size):
    return list(itertools.combinations(items, size))

def find_frequent_itemsets(items, transactions, min_frequency):
    size = 1
    frequent_itemsets = []
    while True:
        itemsets = generate_itemsets(items, size)
        current_frequent_itemsets = [itemset for itemset in itemsets if count_frequency(itemset, transactions) >= min_frequency]
        if not current_frequent_itemsets:
            break
        frequent_itemsets.extend(current_frequent_itemsets)
        size += 1
    return frequent_itemsets

# Generate association rules from the frequent itemsets
def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = tuple(item for item in itemset if item not in antecedent)
                if count_frequency(antecedent, transactions) == 0:
                    continue
                confidence = count_frequency(itemset, transactions) / count_frequency(antecedent, transactions)
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules

def print_sorted_rules(rules):
    sorted_rules = sorted(rules, key=lambda x: x[2], reverse=True)
    for idx, (antecedent, consequent, confidence) in enumerate(sorted_rules, start=1):
        print(f"Rule {idx}: {antecedent} -> {consequent}, Confidence: {confidence:.2f}")

def get_user_choice():
    stores = ['Amazon', 'Best Buy', 'K-Mart', 'Nike', 'Dominos']
    print("Select a store to analyze:")
    for i, store in enumerate(stores, start=1):
        print(f"{i}. {store}")
    choice = int(input("Enter the number corresponding to your choice: ")) - 1
    if 0 <= choice < len(stores):
        selected_store = stores[choice]
        print(f"You selected: {selected_store}")
        return f'{selected_store}_transactions.csv' 
    else:
        print("Invalid choice. Please try again.")
        return None

if __name__ == "__main__":
    filename = get_user_choice()

    if filename:
        transactions = read_transactions(filename)
        unique_items = set(item for transaction in transactions for item in transaction)
        total_transactions = len(transactions)
        support_percentage = float(input("Enter the minimum support threshold (as a percentage, 0-100): "))
        min_frequency = int((support_percentage / 100) * total_transactions)
        min_confidence_percentage = float(input("Enter the minimum confidence threshold (as a percentage, 0-100): "))
        min_confidence = min_confidence_percentage / 100

        # Brute Force Algorithm Execution
        start_time_itemsets = time.perf_counter()
        frequent_itemsets = find_frequent_itemsets(unique_items, transactions, min_frequency)
        brute_force_rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
        end_time_itemsets = time.perf_counter()



        if brute_force_rules:
            print("\nAssociation Rules (Brute Force):")
            print_sorted_rules(brute_force_rules)
        else:
            print("No association rules generated with the Brute Force method.")

        # Applying the Apriori algorithm
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        min_support = min_frequency / total_transactions
        start_time_apriori = time.perf_counter()
        frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)
        apriori_rules = generate_association_rules(frequent_itemsets_apriori['itemsets'].tolist(), transactions, min_confidence)
        end_time_apriori = time.perf_counter()



        if apriori_rules:
            print("\nAssociation Rules (Apriori):")
            print_sorted_rules(apriori_rules)
        else:
            print("No association rules generated with the Apriori method.")

        # Applying the FP-Growth algorithm
        start_time_fpgrowth = time.perf_counter()
        frequent_itemsets_fpgrowth = fpgrowth(df, min_support=min_support, use_colnames=True)
        fpgrowth_rules = generate_association_rules(frequent_itemsets_fpgrowth['itemsets'].tolist(), transactions, min_confidence)
        end_time_fpgrowth = time.perf_counter()



        if fpgrowth_rules:
            print("\nAssociation Rules (FP-Growth):")
            print_sorted_rules(fpgrowth_rules)
        else:
            print("No association rules generated with the FP-Growth method.")

        print(f"\nTime taken for FP-Growth: {end_time_fpgrowth - start_time_fpgrowth:.6f} seconds")
        print(f"Number of frequent itemsets generated (FP-Growth): {len(frequent_itemsets_fpgrowth)}")
        print(f"Number of rules generated (FP-Growth): {len(fpgrowth_rules)}")

        print(f"\nTime taken for Apriori: {end_time_apriori - start_time_apriori:.6f} seconds")
        print(f"Number of frequent itemsets generated (Apriori): {len(frequent_itemsets_apriori)}")
        print(f"Number of rules generated (Apriori): {len(apriori_rules)}")

        print(f"\nTime taken for Brute force: {end_time_itemsets - start_time_itemsets:.6f} seconds")
        print(f"Number of frequent itemsets generated (Brute force): {len(frequent_itemsets)}")
        print(f"Number of association rules generated: {len(brute_force_rules)}")


# ## Conclusion:
# 
# All three algorithms produced the same outputs in terms of the number of frequent itemsets and association rules generated. However, FP-Growth outperformed the others in terms of execution time, making it the most efficient choice for this dataset. The Brute Force method also demonstrated relatively quick performance, but both it and Apriori were slower than FP-Growth.
# This consistency in results across the algorithms indicates the reliability of the findings, while the faster execution of FP-Growth suggests that it is a preferable method for future analyses involving larger datasets.
# 
