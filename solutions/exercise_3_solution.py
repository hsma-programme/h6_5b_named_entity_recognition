# import spacy and the downloaded en_core_web_sm pre-trained model
import spacy
import en_core_web_sm
import en_core_web_md
import en_core_web_lg
import spacy.displacy

# import csv library
import csv

# import tqdm - a library that allows us to use progress bars
# tqdm is in the nlp_ner environment - it comes with spacy
from tqdm import tqdm, trange

# import random library
import random

# Create list to store the headlines in the csv
list_of_headlines = []

with open("abcnews-date-text.csv", "r") as f:
    # Create csv reader object
    reader = csv.reader(f, delimiter=",")

    # For each row of the csv, grab the second element (the headline) and store
    # in the list of headlines
    for row in reader:
        list_of_headlines.append(row[1])

# Remove the first line element from the list of headlines (which was the
# column header)
list_of_headlines.pop(0)

# Load the pre-trained model as a language model into a variable called nlp
# Different sized models available as options for comparison.  Remember any
# models must be downloaded in the Ananconda Prompt / Terminal using
# python -m spacy download name_of_model
# and must be imported into the code
nlp = en_core_web_sm.load()
#nlp = en_core_web_md.load()
#nlp = en_core_web_lg.load()

# Create a list to hold the doc objects
list_of_doc_objects = []

# Randomly shuffle the list of headlines before we sample them
random.shuffle(list_of_headlines)

# Set up the proportion of total headlines in the data that will be analysed
# Note - you may want to play with this depending on the speed of your
# computer
scalar_headlines = 0.01

# Apply the loaded pre-trained SpaCy model to the random proportion of headlines
# defined above, creating a list of doc objects
# We use trange here (from tqdm) instead of range to create a progress bar as
# we move through the loop applying the model.  This will also help you assess
# the scalar value above to use (if it's taking too long, use a smaller
# proportion)
for x in trange(int(len(list_of_headlines) * scalar_headlines)):
    list_of_doc_objects.append(nlp(list_of_headlines[x]))

# Specify the number of random doc objects to pick for displaCy visualisation
number_of_random_docs = 100

# Randomly sample the number of doc objects specified above
random_docs = random.sample(list_of_doc_objects,
                            number_of_random_docs)

# Visualise the predicted Named Entities in the randomly sampled doc objects
# above
for doc_object in random_docs:
    spacy.displacy.render(doc_object, style="ent")

# Function to assemble a dictionary entry for a passed in dictionary, which,
# for a given NER label (passed in), has a list of the predicted named entities
# that are predicted to be of that NER label (type) for a passed in list of
# doc objects
def assemble_dictionary_entry(dictionary, ner_label, list_of_doc_objects):
    # Create empty list of named entities
    list_of_ents = []

    # For each doc object, look through each predicted named entity for that
    # doc object, and add it to the list of entities being assembled if the
    # entity's label matches the one passed in to the function
    for doc_object in list_of_doc_objects:
        for ent in doc_object.ents:
            if ent.label_ == ner_label:
                list_of_ents.append(ent)

    # Create the dictionary entry with the NER label as the key, and the list
    # of entities matching that label as the value
    dictionary[ner_label] = list_of_ents

    # Return the dictionary
    return dictionary

# Create an empty dictionary, which will store named entity labels as keys, and
# lists of the predicted entities matching each label as the values.
ner_dict = {}

# Set up list of named entity labels (this is based off the SpaCy default list)
list_of_ner_types = ["PERSON",
                     "NORP",
                     "FAC",
                     "ORG",
                     "GPE",
                     "LOC",
                     "PRODUCT",
                     "EVENT",
                     "WORK OF ART",
                     "LAW",
                     "LANGUAGE",
                     "DATE",
                     "TIME",
                     "PERCENT",
                     "MONEY",
                     "QUANTITY",
                     "ORDINAL",
                     "CARDINAL"]

# For each label set up in our list above, apply the assemble_dictionary_entry
# function written above to create a dictionary entry with the corresponding
# predicted entities for that label
for type in list_of_ner_types:
    assemble_dictionary_entry(ner_dict, type, list_of_doc_objects)

# Print the dictionary entry showing all the predicted named entities of type
# LOC
print ("Predicted Named Entities of type LOC : ")
print (ner_dict["LOC"])

