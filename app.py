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
    'according', 'told', 'reuters', 'says', 'guardian',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'week', 'month'
}

# Setup
load_dotenv()
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
if not GUARDIAN_API_KEY:
    logger.error("⚠️ No GUARDIAN_API_KEY found in environment!")
    raise ValueError("⚠️ No GUARDIAN_API_KEY found in environment!")

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(CUSTOM_STOP_WORDS)

# Initialize
guardian = GuardianFetcher(GUARDIAN_API_KEY)
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.config.suppress_callback_exceptions = True

@lru_cache(maxsize=32)
def process_articles(start_date, end_date):
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date}")
        df = guardian.fetch_articles(days_back=7, page_size=50)
        
        if df.empty:
            logger.warning("No articles fetched!")
            return None, None, None, None, None
            
        texts = df['content'].apply(lambda x: [
            word.lower() for word in word_tokenize(str(x))
            if word.isalnum() and word.lower() not in stop_words
        ])
        
        if len(texts) < 5:
            logger.warning("Not enough articles for analysis!")
            return None, None, None, None, None
        
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=5,
            id2word=dictionary,
            passes=15,
            random_state=42
        )
        
        return df, texts, dictionary, corpus, lda_model
        
    except Exception as e:
        logger.error(f"Error in process_articles: {str(e)}", exc_info=True)
        return None, None, None, None, None

def create_word_cloud(topic_words):
    try:
        wc = WordCloud(
            background_color='black',
            width=800,
            height=400,
            colormap='viridis'
        ).generate_from_frequencies(dict(topic_words))
        
        fig = px.imshow(wc)
        fig.update_layout(
            template='plotly_dark',
            title="Topic Word Cloud"
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {str(e)}", exc_info=True)
        return go.Figure().update_layout(template='plotly_dark')

def create_tsne_visualization(corpus, lda_model, df):
    try:
        doc_topics = []
        for doc in corpus:
            topic_weights = [0] * lda_model.num_topics
            for topic, weight in lda_model[doc]:
                topic_weights[topic] = weight
            doc_topics.append(topic_weights)
        
        tsne = TSNE(n_components=2, random_state=42)
        topic_coords = tsne.fit_transform(doc_topics)
        
        df_viz = pd.DataFrame({
            'x': topic_coords[:, 0],
            'y': topic_coords[:, 1],
            'topic': [doc_topics[i].index(max(doc_topics[i])) + 1 for i in range(len(doc_topics))],
            'title': df['title']
        })
        
        fig = px.scatter(
            df_viz,
            x='x',
            y='y',
            color='topic',
            hover_data=['title'],
            title='t-SNE Topic Clustering'
        )
        
        fig.update_layout(template='plotly_dark')
        return fig
    except Exception as e:
        logger.error(f"Error creating t-SNE visualization: {str(e)}", exc_info=True)
        return go.Figure().update_layout(template='plotly_dark')

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Guardian News Topic Explorer 📰", className="text-center mb-4"),
            html.P("Interactive analysis of Guardian articles using LDA topic modeling", 
                  className="text-center text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Controls", className="card-title"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=(datetime.now() - timedelta(days=7)).date(),
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
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='topic-distribution')
        ], width=6),
        dbc.Col([
            dcc.Graph(id='word-cloud')
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='tsne-plot')
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
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
    ])
])

@app.callback(
    [Output('topic-distribution', 'figure'),
     Output('word-cloud', 'figure'),
     Output('tsne-plot', 'figure'),
     Output('article-details', 'data')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('topic-filter', 'value')]
)
def update_visualizations(start_date, end_date, selected_topics):
    try:
        logger.info(f"Starting visualization update with dates: {start_date} to {end_date}")
        results = process_articles(start_date, end_date)
        
        if results[0] is None:
            logger.error("Failed to process articles")
            raise ValueError("Failed to process articles")
            
        df, texts, dictionary, corpus, lda_model = results
        
        # Topic Distribution
        topic_terms = []
        for topic_id in range(lda_model.num_topics):
            topic_terms.extend([(word, prob, topic_id) 
                              for word, prob in lda_model.show_topic(topic_id, topn=10)])
        
        topic_df = pd.DataFrame(topic_terms, columns=['word', 'probability', 'topic'])
        
        dist_fig = px.bar(
            topic_df,
            x='probability',
            y='word',
            color='topic',
            orientation='h',
            title='Topic Word Distributions'
        )
        dist_fig.update_layout(template='plotly_dark')
        
        # Word Cloud
        selected_topic = 0 if not selected_topics else selected_topics[0]
        word_cloud_fig = create_word_cloud(lda_model.show_topic(selected_topic, topn=30))
        
        # t-SNE
        tsne_fig = create_tsne_visualization(corpus, lda_model, df)
        
        # Article Details
        doc_topics = []
        for doc in corpus:
            topic_dist = lda_model.get_document_topics(doc)
            doc_topics.append([
                f"Topic {topic+1}: {prob:.3f}"
                for topic, prob in sorted(topic_dist, key=lambda x: x[1], reverse=True)
            ])
        
        articles_data = [{
            'title': row['title'],
            'section': row['section'],
            'published': row['published'].strftime('%Y-%m-%d %H:%M'),
            'topics': '\n'.join(topics)
        } for row, topics in zip(df.to_dict('records'), doc_topics)]
        
        logger.info("Successfully updated all visualizations")
        return dist_fig, word_cloud_fig, tsne_fig, articles_data
        
    except Exception as e:
        logger.error(f"Main callback error: {str(e)}", exc_info=True)
        empty_fig = go.Figure().update_layout(template='plotly_dark')
        return empty_fig, empty_fig, empty_fig, []

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
