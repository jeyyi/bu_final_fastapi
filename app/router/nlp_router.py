from fastapi import APIRouter, HTTPException, UploadFile, File
from wordcloud import WordCloud
from nltk.corpus import stopwords
from typing import List
import itertools
import collections
import pandas as pd
import io
from gensim import corpora
import matplotlib.pyplot as plt
from gensim.models.ldamodel import LdaModel
from collections import Counter
import re
import networkx as nx
from nltk import bigrams
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import app.resources.tgstopwords as tgstopwords


uploaded_stopwords = set()

class TextsRequest(BaseModel):
    texts: List[str]


router_nlp = APIRouter()
def clean_text(text):
    # Convert to lowercase
    text = str(text).lower()

    # Remove all non a-z0-9 characters using regex
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Remove stop words
    stop_words = (stopwords.words('english'))
    stop_words.extend(tgstopwords.generate_tgwords())
    stop_words = set(stop_words)
    # Add uploaded stopwords
    stop_words.update(uploaded_stopwords)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]

    # Join the cleaned words back into a string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text



def get_text(series):
    series = series.dropna()
    series = series.apply(lambda x: clean_text(x))
    return series.tolist()

@router_nlp.post("/upload_stopwords")
async def upload_stopwords(file: UploadFile = File(...)):
    global uploaded_stopwords
    content = await file.read()
    uploaded_stopwords = set(content.decode("utf-8").splitlines())
    return {"message": "Stopwords uploaded successfully!"}

@router_nlp.post("/reset_stopwords")
async def reset_stopwords():
    global uploaded_stopwords
    uploaded_stopwords = set()
    return {"message": "stopwords reset"}

@router_nlp.post("/get_topics/")
def get_topics(request_data: TextsRequest, num_topics=5):
    texts = request_data.texts
    # Create a dictionary representation of the documents
    texts = [clean_text(text) for text in texts]
    tokenized_texts = [text.split() for text in texts if text]
    dictionary = corpora.Dictionary(tokenized_texts)

    # Convert the list of texts to a list of vectors based on word frequency
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics,
                         id2word=dictionary, passes=15)
    # Adjust num_words to display more or fewer keywords per topic
    topics = lda_model.print_topics(num_words=5)
    topics_as_word_lists = []
    num_words_per_topic = 5
    for topic_id in range(lda_model.num_topics):
        topic_word_ids = [word_id for word_id, prob in lda_model.get_topic_terms(
            topic_id, num_words_per_topic)]
        topic_words = [dictionary[word_id] for word_id in topic_word_ids]
        topics_as_word_lists.append(topic_words)

    # return topics_as_word_lists
    return topics_as_word_lists

@router_nlp.post("/get_sentiment")
def get_sentiment(request_data: TextsRequest):
    num_pos = 0
    num_neg = 0
    for text in request_data.texts:
        if get_sentiment_single(text) == "Positive":
            num_pos = num_pos+1
        else:
            num_neg = num_neg+1
    return {"Positive": num_pos, "Negative": num_neg}

def get_sentiment_single(text: str):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    # Interpret the sentiment scores
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return  "Negative"
    else:
        return "Neutral"

@router_nlp.post("/get_frequent/")
def get_frequent(request_data : TextsRequest):
    num_words = 10
    texts = request_data.texts
    # Flatten the list of lists into a single list of words
    texts = [clean_text(text) for text in texts]
    all_words = [word for text in texts for word in text.split()]
    # all_words = [word for word in texts.split()]
    # Use Counter to get word frequencies
    word_counts = Counter(all_words)

    # Return the top 'num_words' frequent words
    return dict(word_counts.most_common(num_words))


@router_nlp.post("/generate_wordcloud")
def generate_wordcloud(request_data: TextsRequest):
    texts = request_data.texts
    texts = [clean_text(text) for text in texts]
    text = " ".join(texts)
    
    # Create wordcloud object
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    # Create an in-memory buffer to store the image
    img_buffer = io.BytesIO()
    
    # Save wordcloud image to buffer
    plt.switch_backend("Agg")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img_buffer, format='png')
    plt.close()
    
    # Seek to the beginning of the buffer
    img_buffer.seek(0)

    # Return the image as a streaming response
    return StreamingResponse(io.BytesIO(img_buffer.read()), media_type="image/png")




@router_nlp.post("/generate_bigramnetwork")
async def generate_bigram_network(request_data: TextsRequest):
    from nltk import bigrams
    texts = request_data.texts
    texts = [clean_text(text) for text in texts]
    texts = [text.split() for text in texts]
    terms_bigrams = [list(bigrams(text)) for text in texts]
    bigrams = list(itertools.chain(*terms_bigrams))
    bigrams_counts = collections.Counter(bigrams)
    bigram_df = pd.DataFrame(bigrams_counts.most_common(20),
                             columns=['bigram', 'count'])
    d = bigram_df.set_index('bigram').T.to_dict('records')
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    img_buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2)

    # Plot networks
    nx.draw_networkx(
        G,
        pos,
        font_size=16,
        width=3,
        edge_color='grey',
        node_color='purple',
        with_labels=False,
        ax=ax
    )

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0] + .135, value[1] + .045
        ax.text(
            x,
            y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center',
            fontsize=13
        )

    # Save the plot to the image buffer
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Define a function to generate the image in chunks
    def generate():
        yield img_buffer.getvalue()

    # Return a StreamingResponse
    return StreamingResponse(
        generate(),
        media_type="image/png"
    )