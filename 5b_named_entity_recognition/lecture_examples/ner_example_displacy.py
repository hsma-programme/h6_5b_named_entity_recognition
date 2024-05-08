# import spacy and the downloaded en_core_web_sm pre-trained model
import spacy
import en_core_web_sm
import spacy.displacy

# Load the pre-trained model as a language model into a variable called nlp
nlp = en_core_web_sm.load()

# Read in the document for which we want to extract named entities
with open("titchmarsh.txt", encoding='utf8') as f:
    # Read in the whole file as a single string but strip out any whitespace at
    # the start and end of the file
    raw_read = f.read().strip()
    
# Apply the pre-trained model to the raw text string to extract named entities
# All of SpaCy's predictions (including the named entity predictions) are
# stored as a Doc object.  Here, we call our Doc object 'doc', but you
# may also see it named 'article'.
doc = nlp(raw_read)

# Render named entities using displacy
spacy.displacy.render(doc, style="ent")

