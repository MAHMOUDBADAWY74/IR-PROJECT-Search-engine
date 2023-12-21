import os
import math
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.stem import PorterStemmer

# First part
files = natsorted(os.listdir('files'))

print("------------------------------files-------------------------------")
print(files)

print("------------------------------document-------------------------------")

tokenized_documents = []

for file in files:
    with open(f"files/{file}") as f:
        document = f.read()
        print(document)

        tokenized_document = word_tokenize(document)
        list_of_terms = []
        for term in tokenized_document:
                list_of_terms.append(term)
        tokenized_documents.append(list_of_terms)

print("------------------------------tokenized_document-------------------------------")
print(tokenized_documents)

# Second phase with stemming
def stemming(tokenized_documents):
    stemmer = PorterStemmer()
    stem_document = []

    for terms in tokenized_documents:
        stemmed_terms = []  # Updated variable name
        for word in terms:
            stemmed_word = stemmer.stem(word)
            stemmed_terms.append(stemmed_word)
        stem_document.append(stemmed_terms)

    return stem_document

stem = stemming(tokenized_documents)
print("--------------------------stemming-----------------------------------")
print(stem)

# Creating positional indices list
documents_number = 0
positional_indices_list = {}

for stemmed_document in stem:
    for position, term in enumerate(stemmed_document):
        if term in positional_indices_list:
            positional_indices_list[term][0] += 1
            if documents_number in positional_indices_list[term][1]:
                positional_indices_list[term][1][documents_number].append(position)
            else:
                positional_indices_list[term][1][documents_number] = [position]
        else:
            positional_indices_list[term] = []
            positional_indices_list[term].append(1)
            positional_indices_list[term].append({})
            positional_indices_list[term][1][documents_number] = [position]
    documents_number += 1

print("------------------------------Positional indices-------------------------------")
print(positional_indices_list)

# Third phase
one_document_to_contain_them_all = []

for document in stem:
    for token in document:
        one_document_to_contain_them_all.append(token)

# Getting term frequencies
def get_terms_frequencies_in_a_document(document):
    terms_frequencies = dict.fromkeys(one_document_to_contain_them_all, 0)
    for term in document:
        terms_frequencies[term] += 1
    return terms_frequencies

def get_weighted_terms_frequencies_in_a_document(term_frequency):
    if term_frequency > 0:
        return math.log(term_frequency) + 1
    return 0

def get_term_frequencies():
    terms_frequencies = pd.DataFrame()
    for i in range(len(stem)):
        terms_frequencies[i] = get_terms_frequencies_in_a_document(stem[i])
    return terms_frequencies

terms_frequencies = get_term_frequencies()

# Rename the columns in the DataFrame
new_column_names = [f"Doc{i+1}" for i in range(len(stem))]
terms_frequencies.columns = new_column_names

print("------------------------------Terms frequencies with renamed columns-------------------------------")
print(terms_frequencies)

# Getting weighted_terms_frequencies
def get_weighted_terms_frequencies():
    weighted_terms_frequencies = pd.DataFrame()
    for i in range(len(stem)):
        column_name = new_column_names[i]
        weighted_terms_frequencies[column_name] = terms_frequencies[column_name].apply(get_weighted_terms_frequencies_in_a_document)
    return weighted_terms_frequencies

weighted_terms_frequencies = get_weighted_terms_frequencies()
print("------------------------------weighted terms frequencies-------------------------------")
print(weighted_terms_frequencies)

# Getting idf
idf_data_frame = pd.DataFrame(columns=['frequency', 'idf'])

for i in range(len(terms_frequencies)):
    frequency = terms_frequencies.iloc[i].values.sum()
    idf_data_frame.loc[i, 'frequency'] = frequency
    idf_data_frame.loc[i, 'idf'] = math.log10(documents_number / float(frequency))


idf_data_frame.index = terms_frequencies.index
print("------------------------------df_idf-------------------------------")
print(idf_data_frame)

# Getting tf*idf
tf_multiply_idf = terms_frequencies.multiply(idf_data_frame['idf'], axis=0)

print("------------------------------tf*idf-------------------------------")
print(tf_multiply_idf)

# Getting Documents_lenghts
def get_document_length(document):
    return np.sqrt(tf_multiply_idf[document].apply(lambda x: x**2).sum())

documents_lenghts = pd.DataFrame()
for document in tf_multiply_idf.columns:
    documents_lenghts.loc[0, document] = get_document_length(document)

print("------------------------------Documents_lenghts-------------------------------")
print(documents_lenghts)


# Getting Normlized tf*idf
normlized_tf_idf = pd.DataFrame()

def normlize(document, tf_idf):
    try:
        return tf_idf / documents_lenghts[document].values[0]
    except ZeroDivisionError:
        return 0
    
for document in tf_multiply_idf.columns:
    normlized_tf_idf[document] = tf_multiply_idf[document].apply(lambda tf_idf: normlize(document, tf_idf))

print("------------------------------Normlized tf*idf-------------------------------")
print(normlized_tf_idf)

# Function to tokenize and stem a query
def tokenize_and_stem(query):
    stemmer = PorterStemmer()
    tokens = word_tokenize(query)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Making queries
def get_query_respond(query, query_terms):
    respond_text = 'Matches are found in: '
    occurrences_list = [[] for _ in range(documents_number)]

    # Processing each term in the query
    for term in query_terms:
        try:
            # Building a list of occurrences for each document
            for key in positional_indices_list[term][1].keys():
                if occurrences_list[key] != []:
                    if occurrences_list[key][-1] == positional_indices_list[term][1][key][0] - 1:
                        occurrences_list[key].append(positional_indices_list[term][1][key][0])
                    else:
                        occurrences_list[key] = [positional_indices_list[term][1][key][0]]
                else:
                    occurrences_list[key].append(positional_indices_list[term][1][key][0])

            # Checking if there is a match for the entire query in a document
            for position, occurrence in enumerate(occurrences_list, start=1):
                if len(occurrence) == len(query_terms):
                    respond_text = respond_text + f'doc{position} '
        except KeyError:
            print(f"Term \"{term}\" is not found!")

    return respond_text

# Checking if there are matches for the query
def is_there_match(respond_text):
    return len(respond_text) > 21

# Building a DataFrame for the query response
query_respond = pd.DataFrame()
query_respond.index = normlized_tf_idf.index

# Getting the tokenized and stemmed query
query = input('What do you want to find?\n').lower()
query_terms = tokenize_and_stem(query)
query_respond['tf'] = [1 if term in query_terms else 0 for term in normlized_tf_idf.index]
# Getting weighted term frequencies for the query
def get_query_weighted_tf(tf):
    return math.log(tf) + 1 if tf > 0 else 0

# Processing the query response
respond = get_query_respond(query, query_terms)
if is_there_match(respond):
    query_respond['wtf'] = query_respond['tf'].apply(get_query_weighted_tf)
    query_respond['idf'] = idf_data_frame['idf'] * query_respond['wtf']
    query_respond['tf_idf'] = query_respond['idf'] * query_respond['wtf']
    query_respond['normalized'] = 0

    # Normalizing the query response
    for i in range(len(query_respond)):
        query_respond['normalized'].iloc[i] = float(query_respond['idf'].iloc[i]) / math.sqrt(
            sum(query_respond['idf'].values ** 2))

    # Calculating product1 and product2
    product1 = normlized_tf_idf.multiply(query_respond['wtf'], axis=0)
    product2 = product1.multiply(query_respond['normalized'], axis=0)

    scores = {}

    # Calculating the cosine similarity scores
    for column in product2.columns:
        if 0 in product2[column].loc[query_terms].values:
            pass
        else:
            scores[column] = product2[column].sum()

    # Printing the query response and scores
    print(query_respond)
    print(scores)
    print(product2[list(scores.keys())])
    print(respond)
else:
    print("No matches are found!")


# Function to process queries and return relevant document
def returned(query, documents, normlized_tf_idf, idf_data_frame):
    stemmer = PorterStemmer()

    # Tokenizing and stemming the query
    query_terms = [stemmer.stem(token) for token in word_tokenize(query)]

    query_df = pd.DataFrame(index=normlized_tf_idf.index)
    query_df['tf'] = [1 if term in query_terms else 0 for term in normlized_tf_idf.index]

    # Function to calculate weighted term frequencies for the query
    def get_w_tf(x):
        try:
            return math.log10(x) + 1
        except:
            return 0

    # Normalizing the query response
    query_df['w_tf'] = query_df['tf'].apply(lambda x: get_w_tf(x))
    query_df['idf'] = idf_data_frame['idf'] * query_df['w_tf']
    query_df['tf_idf'] = query_df['w_tf'] * query_df['idf']
    query_df['norm'] = 0

    # Normalizing again (seems redundant, please check if necessary)
    for i in range(len(query_df)):
        query_df['norm'].iloc[i] = float(query_df['idf'].iloc[i]) / math.sqrt(sum(query_df['idf'].values ** 2))

    print("\n\n\n query")
    print(query_df)

    product = normlized_tf_idf.multiply(query_df['w_tf'], axis=0)
    query_df['norm'] = 0

    for i in range(len(query_df)):
        query_df['norm'].iloc[i] = float(query_df['idf'].iloc[i]) / math.sqrt(sum(query_df['idf'].values ** 2))

    product2 = product.multiply(query_df['norm'], axis=0)

    print("\n\nquery length")
    print(math.sqrt(sum(x ** 2 for x in query_df['idf'].loc[query_terms])))
    print("\n\n\n products")

    math.sqrt(sum(x ** 2 for x in query_df['idf'].loc[query_terms]))
    scores = {}

    for col in product2.columns:
        if 0 in product2[col].loc[query_terms].values:
            pass
        else:
            scores[col] = product2[col].sum()

    prod_res = product2[list(scores.keys())].loc[query_terms]
    print(prod_res)
    print("\n\nsum of doc")
    print(prod_res.sum())

    final_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n\ncosine similarity")
    for doc in final_score:
        print(doc, end=' ')

    print("\n\n returned docs\n\n\n")

    for doc in final_score:
        print(doc[0], end=' ')
        print('\n')


# Assuming 'files' is the list of file names in your 'files' directory
files = natsorted(os.listdir('files'))

# Assuming 'tokenized_documents' is the list of tokenized documents
tokenized_documents = []

for file in files:
    with open(f"files/{file}") as f:
        document = f.read()
        tokenized_document = [stemming(token) for token in word_tokenize(document)]
        tokenized_documents.append(tokenized_document)

# ... (your previous code)

a = input("\n\nenter query ")
returned(a, tokenized_documents, normlized_tf_idf, idf_data_frame)
quer = input('input your query to need to know pos_index:')
print(get_query_respond(quer))