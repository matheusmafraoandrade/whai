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
data_load_state.text("")

#basket.replace({'brandy':'Pão Italiano', 'softener':'Vela', 'canned fruit':'Queijo',
#                'syrup':'Chocolate', 'artif. sweetener':'Morango', 'whole milk':'Vinho'}, inplace=True)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(basket)

st.sidebar.title("Whai")

### Basket -> Transactions
def transform_into_transactions(df):
    df.itemDescription = df.itemDescription.transform(lambda x: [x])
    df = df.groupby(['Member_number','Date']).sum()['itemDescription'].reset_index(drop=True)
    encoder = TransactionEncoder()
    transactions = pd.DataFrame(encoder.fit(df).transform(df), columns=encoder.columns_)
    return transactions

transactions = transform_into_transactions(basket)

### Association Rules
def mine_itemsets(df_transactions, df_basket):
    min_support = st.sidebar.slider("Suporte mínimo", min_value=0.0, max_value=1.0, value=6/len(basket))
    frequent_itemsets = fpgrowth(df_transactions, min_support=min_support, use_colnames=True, max_len = 2)

    metric = st.sidebar.selectbox("Métrica", ("support", "confidence", "lift"))
    if metric=="support":
        min_threshold=0.5
    elif metric=="confidence":
        min_threshold=0.2
    else:
        min_threshold=1.5

    rules = association_rules(frequent_itemsets, metric=metric,  min_threshold=min_threshold).sort_values(
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

with st.sidebar:
    st.header("Whai")