#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Below is the full, corrected Dash code that:
1) Resets the filtered DataFrame’s index before referencing corpus.
2) Converts doc_topics to a NumPy array for t-SNE.
3) Adjusts perplexity if fewer than 30 documents exist.
4) Avoids "list index out of range" by ensuring df.index matches corpus.

Please replace your existing app.py (or code) with this version.
"""

import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from guardian_fetcher import GuardianFetcher
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import nltk
import os
from dotenv import load_dotenv
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enhanced stop words
CUSTOM_STOP_WORDS = {
    'says', 'said', 'would', 'also', 'one', 'new',
    'us', 'people', 'government', 'could', 'will',
    'may', 'like', 'get', 'make', 'first', 'two',
    'year', 'years', 'time', 'way', 'says', 'trump',
    'according', 'told', 'reuters', 'guardian',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'week', 'month'
}

# Load environment variables
load_dotenv()
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
if not GUARDIAN_API_KEY:
    logger.error("⚠️ No GUARDIAN_API_KEY found in environment!")
    raise ValueError("⚠️ No GUARDIAN_API_KEY found in environment!")

# NLTK tokenizers
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(CUSTOM_STOP_WORDS)

# Initialize GuardianFetcher
guardian = GuardianFetcher(GUARDIAN_API_KEY)

# Initialize Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])
server = app.server
app.config.suppress_callback_exceptions = True

@lru_cache(maxsize=64)
def process_articles(start_date, end_date):
    """
    Fetch and process Guardian articles from start_date to end_date.
    Returns DataFrame, texts, dictionary, corpus, lda_model.
    """
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date}")
        
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        days_back = (datetime.now().date() - start_date_dt).days + 1
        
        # Fetch articles
        df = guardian.fetch_articles(days_back=days_back, page_size=200)
        if df.empty:
            logger.warning("No articles fetched!")
            return None, None, None, None, None
        
        # Filter to exact date range
        df = df[
            (df['published'].dt.date >= start_date_dt) &
            (df['published'].dt.date <= end_date_dt)
        ]
        logger.info(f"Filtered to {len(df)} articles within date range")
        
        # Process text
        texts = []
        for i in df.index:
            content = df.at[i, 'content']
            if pd.isna(content):
                texts.append([])
                continue
            words = word_tokenize(str(content))
            filtered_words = [
                w.lower() for w in words 
                if w.isalnum() and w.lower() not in stop_words
            ]
            texts.append(filtered_words)
        
        if len(df) < 5:
            logger.warning("Not enough articles for analysis!")
            return None, None, None, None, None
        
        # Reset index so df.index will be 0..(len(df)-1)
        df.reset_index(drop=True, inplace=True)
        
        # Build dictionary & corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(t) for t in texts]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=5,
            id2word=dictionary,
            passes=20,
            random_state=42,
            chunksize=100
        )
        
        logger.info(f"Successfully processed {len(df)} articles")
        return df, texts, dictionary, corpus, lda_model
    
    except Exception as e:
        logger.error(f"Error in process_articles: {str(e)}", exc_info=True)
        return None, None, None, None, None

def create_word_cloud(topic_words):
    """
    Create a WordCloud figure from LDA topic word list.
    """
    try:
        freq_dict = dict(topic_words)
        wc = WordCloud(
            background_color='black',
            width=800,
            height=400,
            colormap='viridis'
        ).generate_from_frequencies(freq_dict)
        
        fig = px.imshow(wc)
        fig.update_layout(template='plotly', title="Topic Word Cloud")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {str(e)}", exc_info=True)
        return go.Figure().update_layout(template='plotly')

def create_tsne_visualization(df, corpus, lda_model):
    """
    Create t-SNE scatter from the filtered df and the corpus.
    Expects df.index to match corpus indexing (after df.reset_index).
    """
    try:
        # Build doc_topics from each doc in the corpus, 
        # using df.index as the reference
        doc_topics = []
        for i in df.index:  # now i goes from 0..len(df)-1
            topic_weights = [0.0] * lda_model.num_topics
            for t_id, weight in lda_model[corpus[i]]:
                topic_weights[t_id] = weight
            doc_topics.append(topic_weights)
        
        doc_topics_array = np.array(doc_topics, dtype=np.float32)
        
        # If fewer than 2 docs, skip TSNE
        if doc_topics_array.shape[0] < 2:
            fig = go.Figure().update_layout(template='plotly', 
                                            title='Not enough documents for t-SNE')
            return fig
        
        # Safely adjust perplexity
        perplex_val = 30
        if doc_topics_array.shape[0] < 30:
            perplex_val = max(2, doc_topics_array.shape[0] - 1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplex_val)
        topics_2d = tsne.fit_transform(doc_topics_array)
        
        temp_df = pd.DataFrame({
            'x': topics_2d[:, 0],
            'y': topics_2d[:, 1],
            # +1 for a human-readable topic label
            'dominant_topic': [np.argmax(doc_topics[i]) + 1 for i in range(len(doc_topics))],
            'doc_index': df.index,  # store actual index
            'title': df['title']
        })
        
        fig = px.scatter(
            temp_df,
            x='x',
            y='y',
            color='dominant_topic',
            hover_data=['title'],
            title='t-SNE Topic Clustering',
            custom_data=['doc_index']
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating t-SNE visualization: {str(e)}", exc_info=True)
        fig = go.Figure().update_layout(template='plotly', title=f"t-SNE Error: {e}")
        return fig

###########################################
# LAYOUT
###########################################

navbar = dbc.NavbarSimple(
    brand="Guardian News Topic Explorer",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Example banner (replacing the old Jumbotron)
banner = dbc.Container(
    [
        html.H1("Guardian News Topic Explorer 📰", className="display-3 fw-bold"),
        html.P(
            "Interactive topic modeling and t-SNE clustering of Guardian articles, powered by LDA.",
            className="lead text-muted"
        ),
    ],
    fluid=True,
    className="py-5 my-4 bg-light rounded-3 text-center"
)

controls_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Controls")),
        dbc.CardBody([
            dbc.RadioItems(
                id='date-select-buttons',
                options=[
                    {'label': 'Last Day', 'value': 'last_day'},
                    {'label': 'Last Week', 'value': 'last_week'},
                    {'label': 'Last Month', 'value': 'last_month'},
                ],
                value='last_month',
                inline=True,
                className="mb-3",
            ),
            dcc.DatePickerRange(
                id='date-range',
                start_date=(datetime.now() - timedelta(days=30)).date(),
                end_date=datetime.now().date(),
                className="mb-3"
            ),
            dcc.Dropdown(
                id='topic-filter',
                options=[{'label': f'Topic {i+1}', 'value': i} for i in range(5)],
                multi=True,
                placeholder="Filter by topics...",
                className="mb-3"
            )
        ])
    ],
    className="mb-4 shadow"
)

topic_distribution_graph = dbc.Card(
    [
        dbc.CardHeader("Topic Word Distributions", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='topic-distribution', style={"height": "400px"})
        ])
    ],
    className="mb-4 shadow"
)

word_cloud_graph = dbc.Card(
    [
        dbc.CardHeader("Word Cloud", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='word-cloud', style={"height": "400px"})
        ])
    ],
    className="mb-4 shadow"
)

tsne_plot_graph = dbc.Card(
    [
        dbc.CardHeader("t-SNE Topic Clustering", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='tsne-plot', style={"height": "600px"})
        ])
    ],
    className="mb-4 shadow"
)

articles_table = dbc.Card(
    [
        dbc.CardHeader("Article Details", className="bg-secondary text-light"),
        dbc.CardBody([
            dash_table.DataTable(
                id='article-details',
                columns=[
                    {'name': 'Title', 'id': 'title'},
                    {'name': 'Section', 'id': 'section'},
                    {'name': 'Published', 'id': 'published'},
                    {'name': 'Topics', 'id': 'topics'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white',
                    'textAlign': 'left'
                },
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'fontWeight': 'bold'
                },
                page_size=10
            )
        ])
    ],
    className="mb-4 shadow"
)

app.layout = dbc.Container(
    [
        navbar,
        banner,
        dbc.Row([
            dbc.Col(controls_card, md=4),
            dbc.Col([
                topic_distribution_graph,
                word_cloud_graph
            ], md=8)
        ], align="start"),
        dbc.Row([dbc.Col(tsne_plot_graph, md=12)]),
        dbc.Row([dbc.Col(articles_table, md=12)])
    ],
    fluid=True
)

###########################################
# CALLBACKS
###########################################

@app.callback(
    [Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    [Input('date-select-buttons', 'value')]
)
def update_date_range(selected_range):
    end_date = datetime.now().date()
    if selected_range == 'last_day':
        start_date = end_date - timedelta(days=1)
    elif selected_range == 'last_week':
        start_date = end_date - timedelta(days=7)
    elif selected_range == 'last_month':
        start_date = end_date - timedelta(days=30)
    else:
        start_date = end_date - timedelta(days=30)
    return start_date, end_date

@app.callback(
    [
        Output('topic-distribution', 'figure'),
        Output('word-cloud', 'figure'),
        Output('tsne-plot', 'figure'),
        Output('article-details', 'data')
    ],
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('topic-filter', 'value')
    ]
)
def update_visualizations(start_date, end_date, selected_topics):
    """
    1) Fetch and process data.
    2) Topic bar chart.
    3) Word cloud for first selected topic.
    4) t-SNE scatter (using create_tsne_visualization).
    5) Article table with topic membership.
    """
    try:
        logger.info(f"Starting update with dates: {start_date} to {end_date}")
        results = process_articles(start_date, end_date)
        if not results or results[0] is None:
            logger.error("Failed to process articles")
            empty_fig = go.Figure().update_layout(template='plotly')
            return empty_fig, empty_fig, empty_fig, []
        
        df, texts, dictionary, corpus, lda_model = results
        
        # If no topics selected, show all
        if not selected_topics:
            selected_topics = list(range(lda_model.num_topics))
        
        # Build topic distribution figure
        # gather top words from each selected topic
        topic_terms = []
        for topic_id in selected_topics:
            top_words = lda_model.show_topic(topic_id, topn=10)
            for word, prob in top_words:
                topic_terms.append((word, prob, topic_id))
        
        topic_df = pd.DataFrame(topic_terms, columns=['word', 'probability', 'topic'])
        dist_fig = px.bar(
            topic_df,
            x='probability',
            y='word',
            color='topic',
            orientation='h',
            title='Topic Word Distributions'
        )
        dist_fig.update_layout(template='plotly')
        
        # Word Cloud for the first selected topic
        sel_topic = selected_topics[0] if selected_topics else 0
        wcloud_fig = create_word_cloud(lda_model.show_topic(sel_topic, topn=30))
        
        # t-SNE
        tsne_fig = create_tsne_visualization(df, corpus, lda_model)
        
        # Prepare article data
        # For each doc in df, gather topic distribution
        table_data = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            # keep only selected topics in textual representation
            topic_info = []
            for t_id, prob in sorted(doc_topics, key=lambda x: x[1], reverse=True):
                if t_id in selected_topics:
                    topic_info.append(f"Topic {t_id+1}: {prob:.3f}")
            
            row_data = {
                'title': df.at[i, 'title'],
                'section': df.at[i, 'section'],
                'published': df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(topic_info)
            }
            table_data.append(row_data)
        
        logger.info("Successfully updated all visuals")
        return dist_fig, wcloud_fig, tsne_fig, table_data
    except Exception as e:
        logger.error(f"Main callback error: {str(e)}", exc_info=True)
        empty_fig = go.Figure().update_layout(template='plotly')
        return empty_fig, empty_fig, empty_fig, []

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
