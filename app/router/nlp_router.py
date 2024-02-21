from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
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
import plotly.graph_objects as go


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
    
@router_nlp.post("/get_emots")
def get_emots(request_data:TextsRequest):
    df = tgstopwords.generate_rulebased()
    lst_emotions = []
    texts = " ".join(request_data.texts)
    texts = texts.lower()
    for word in texts.split():
        if word in list(df['Filipino Word']):
            index = df.index[df['Filipino Word'] == word].tolist()[0]
            lst_emotions.append(df.columns[df.iloc[index].eq(1)].tolist())
        if word in list(df['English Word']):
            index = df.index[df['English Word'] == word].tolist()[0]
            lst_emotions.append(df.columns[df.iloc[index].eq(1)].tolist())
    #flatten emotions
    try:
        flat_emotions = [item for sublist in lst_emotions for item in sublist]
        counts = Counter(flat_emotions)
        return counts
    except:
        return []

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

@router_nlp.post("/generate_emotiongraph")
def generate_emotiongraph(request_data: TextsRequest):
    df = tgstopwords.generate_rulebased()
    texts = ' '.join(request_data.texts)
    # Process the text: normalize case, split into words, and count frequency
    words_in_text = texts.lower().split()
    word_frequency = Counter(words_in_text)

    # Filter the dataframe to include only words present in the sample text and in the lexicon
    df_filtered = df[df['Filipino Word'].str.lower().isin(words_in_text)].copy()

    # Calculate sentiment and intensity scores
    # Correct the references to the column names to match the DataFrame's column names
    df_filtered['sentiment_score'] = df_filtered['positive'] - df_filtered['negative']
    df_filtered['intensity_score'] = df_filtered.iloc[:, 1:10].sum(axis=1)

    # Map the frequency from the sample text to the dataframe
    df_filtered['frequency'] = df_filtered['Filipino Word'].str.lower().map(word_frequency)

    # Aggregate the data by each word to ensure they appear only once
    df_aggregated = df_filtered.groupby('Filipino Word').agg({
        'sentiment_score': 'mean',
        'intensity_score': 'mean',
        'frequency': 'sum'
    }).reset_index()

    # Exclude words that are exactly at (0,0)
    df_aggregated = df_aggregated[~((df_aggregated['sentiment_score'] == 0) & (df_aggregated['intensity_score'] == 0))]

    # Normalize the size of the circles
    df_aggregated['size'] = df_aggregated['frequency'].apply(lambda x: x**0.5 * 10)

    # Define colors for each quadrant
    quadrant_colors = {
        'Pleasant-Intense': 'green',
        'Unpleasant-Intense': 'red',
        'Unpleasant-Mild': 'blue',
        'Pleasant-Mild': 'orange'
    }

    median_intensity = df_filtered['intensity_score'].median()

    # Function to adjust intensity score for 'mild' words
    def adjust_intensity_for_mild(intensity, median_intensity):
        # If intensity is below median, it's considered 'mild' and we make it negative
        return -intensity if intensity < median_intensity else intensity

    # Apply the function to adjust the intensity scores
    df_aggregated['adjusted_intensity_score'] = df_aggregated['intensity_score'].apply(
        adjust_intensity_for_mild, args=(median_intensity,))

    # Function to determine the quadrant with the new logic
    def determine_quadrant(sentiment, adjusted_intensity):
        if sentiment > 0 and adjusted_intensity >= 0:
            return 'Pleasant-Intense'
        elif sentiment < 0 and adjusted_intensity >= 0:
            return 'Unpleasant-Intense'
        elif sentiment < 0 and adjusted_intensity < 0:
            return 'Unpleasant-Mild'
        elif sentiment > 0 and adjusted_intensity < 0:
            return 'Pleasant-Mild'

    # Assign quadrant color and name based on sentiment and adjusted intensity score
    df_aggregated['quadrant'] = df_aggregated.apply(
        lambda x: determine_quadrant(x['sentiment_score'], x['adjusted_intensity_score']), axis=1
    )

    # Map colors to the aggregated data
    df_aggregated['color'] = df_aggregated['quadrant'].map(quadrant_colors)

    # Create a scatter plot
    fig = go.Figure()

    # Add the scatter plot for each word
    # Group data by quadrant
    for quadrant, color in quadrant_colors.items():
        quadrant_df = df_aggregated[df_aggregated['quadrant'] == quadrant]
        fig.add_trace(go.Scatter(
            x=quadrant_df['sentiment_score'],
            y=quadrant_df['adjusted_intensity_score'],
            text=quadrant_df['Filipino Word'],
            mode='markers+text',
            marker=dict(
                size=quadrant_df['size'],
                color=color,
                line=dict(width=2, color='DarkSlateGrey'),
                opacity=0.6
            ),
            name=quadrant,
            textposition='bottom center'
        ))


    # Complete the figure layout settings
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='Black'),
        yaxis=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='Black'),
        xaxis_title="Sentiment Score (Unpleasant - Pleasant)",
        yaxis_title="Intensity Score (Mild - Intense)",
        legend=dict(
            title="Quadrants",
            yanchor="bottom",
            y=-0.1,
            xanchor="right",
            x=1.2
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        width=700,  # Set the same value for width and height
        height=550,  # to ensure the plot is square in shape
        autosize=False  # Disable autosize to enforce the specified dimensions
    )

    # Add quadrant lines that extend across the full plot
    max_axis_value = max(df_aggregated['sentiment_score'].abs().max(), df_aggregated['intensity_score'].abs().max())
    fig.add_shape(type="line", x0=0, y0=-max_axis_value, x1=0, y1=max_axis_value,
                line=dict(color="Black", width=2))
    fig.add_shape(type="line", x0=-max_axis_value, y0=0, x1=max_axis_value, y1=0,
                line=dict(color="Black", width=2))
    fig_html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    return fig_html
