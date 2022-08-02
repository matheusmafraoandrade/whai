import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import streamlit as st
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title('Ranking de Associações')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
basket = pd.read_csv(r"Groceries_dataset.csv")
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done!")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(basket)


def transform_into_transactions(df):
    df.itemDescription = df.itemDescription.transform(lambda x: [x])
    df = df.groupby(['Member_number','Date']).sum()['itemDescription'].reset_index(drop=True)
    encoder = TransactionEncoder()
    transactions = pd.DataFrame(encoder.fit(df).transform(df), columns=encoder.columns_)
    return transactions

#st.subheader('Transações')
transactions = transform_into_transactions(basket)
#st.table(transactions)

def mine_itemsets(df_transactions, df_basket):
    frequent_itemsets = fpgrowth(df_transactions, min_support= 6/len(df_basket), use_colnames=True, max_len = 2)
    rules = association_rules(frequent_itemsets, metric="lift",  min_threshold = 1.5).sort_values(
            'confidence', axis=0, ascending=False).reset_index()
    rules['antecedents'] = rules['antecedents'].astype('str').str.replace(r'[^(]*\({|\}\)[^)]*', '')
    rules['consequents'] = rules['consequents'].astype('str').str.replace(r'[^(]*\({|\}\)[^)]*', '')
    return rules

st.subheader('Pares de itens')
itemsets = mine_itemsets(transactions, basket)
st.write(itemsets[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

st.subheader('Top 5 Associações')
col1, col2, col3, col4, col5 = st.columns(5)
for i in range(0,5):
    col1.metric("Antecedente", f"{itemsets['antecedents'][i]}")
    col2.metric("Consequente", f"{itemsets['consequents'][i]}")
    col3.metric("Suporte", f"{(itemsets['support'][i]*100).round(2)}%")
    col4.metric("Confiança", f"{(itemsets['confidence'][i]*100).round(2)}%")
    col5.metric("Lift", f"{(itemsets['lift'][i]).round(2)}")
