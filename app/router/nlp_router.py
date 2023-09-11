from fastapi import APIRouter
from nltk.corpus import stopwords
from typing import List
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from collections import Counter
import re
router_nlp = APIRouter()

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove all non a-z0-9 characters using regex
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    # Join the cleaned words back into a string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def get_text(series):
    series = series.dropna()
    series = series.apply(lambda x: clean_text(x))
    return series.tolist()

@router_nlp.post("/get_topics/")
def get_topics(texts: List[str], num_topics = 5):
    # Create a dictionary representation of the documents
    texts = [clean_text(text) for text in texts]
    tokenized_texts = [text.split() for text in texts if text]
    dictionary = corpora.Dictionary(tokenized_texts)

    # Convert the list of texts to a list of vectors based on word frequency
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)  # Adjust num_words to display more or fewer keywords per topic
    topics_as_word_lists = []
    num_words_per_topic =5
    for topic_id in range(lda_model.num_topics):
        topic_word_ids = [word_id for word_id, prob in lda_model.get_topic_terms(topic_id, num_words_per_topic)]
        topic_words = [dictionary[word_id] for word_id in topic_word_ids]
        topics_as_word_lists.append(topic_words)

    #return topics_as_word_lists
    return topics_as_word_lists

@router_nlp.post("/get_frequent/")
def get_frequent(texts: List[str], num_words: int):
    # Flatten the list of lists into a single list of words
    all_words = [word for text in texts for word in text.split()]
    #all_words = [word for word in texts.split()]
    # Use Counter to get word frequencies
    word_counts = Counter(all_words)

    # Return the top 'num_words' frequent words
    return dict(word_counts.most_common(num_words))

