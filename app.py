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
app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])
server = app.server
app.config.suppress_callback_exceptions = True

@lru_cache(maxsize=64)
def process_articles(start_date, end_date):
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date}")
        
        # Date calculations
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
        
        # Fix the text processing - This was the bug!
        texts = []
        for content in df['content']:
            if pd.isna(content):
                continue
            words = word_tokenize(str(content))
            filtered_words = [word.lower() for word in words 
                            if word.isalnum() and word.lower() not in stop_words]
            texts.append(filtered_words)
        
        if len(texts) < 5:
            logger.warning("Not enough articles for analysis!")
            return None, None, None, None, None
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
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
        
        # Convert list to numpy array before TSNE
        doc_topics_array = np.array(doc_topics)
        
        tsne = TSNE(n_components=2, random_state=42)
        topic_coords = tsne.fit_transform(doc_topics_array)  # Now using numpy array
        
        df_viz = pd.DataFrame({
            'x': topic_coords[:, 0],
            'y': topic_coords[:, 1],
            'topic': [np.argmax(doc_topics[i]) + 1 for i in range(len(doc_topics))],
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
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
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
                        # New date selection buttons
                        dbc.RadioItems(
                            id='date-select-buttons',
                            options=[
                                {'label': 'Last Day', 'value': 'last_day'},
                                {'label': 'Last Week', 'value': 'last_week'},
                                {'label': 'Last Month', 'value': 'last_month'},
                            ],
                            value='last_month',
                            inline=True,
                            className="mb-3"
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
        start_date = end_date - timedelta(days=30)  # default to last month
        
    return start_date, end_date


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
        
        # If no topics selected, show all topics
        if not selected_topics:
            selected_topics = list(range(lda_model.num_topics))
        
        # Topic Distribution - Now filtered by selected topics
        topic_terms = []
        for topic_id in selected_topics:  # Only include selected topics
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
        
        # Word Cloud - Already working with selected topics
        selected_topic = selected_topics[0] if selected_topics else 0
        word_cloud_fig = create_word_cloud(lda_model.show_topic(selected_topic, topn=30))
        
        # t-SNE - Filter by selected topics
        doc_topics = []
        doc_topic_mapping = []  # To track which topics each document belongs to
        
        for doc in corpus:
            topic_weights = [0] * lda_model.num_topics
            for topic, weight in lda_model[doc]:
                topic_weights[topic] = weight
            doc_topics.append(topic_weights)
            
            # Get the dominant topic for this document
            dominant_topic = max(range(len(topic_weights)), key=lambda i: topic_weights[i])
            doc_topic_mapping.append(dominant_topic)
        
        # Filter documents by selected topics
        mask = [idx for idx, topic in enumerate(doc_topic_mapping) if topic in selected_topics]
        filtered_doc_topics = [doc_topics[i] for i in mask]
        filtered_df = df.iloc[mask]
        
        tsne_fig = create_tsne_visualization(
            [corpus[i] for i in mask],  # filtered corpus
            lda_model,
            filtered_df
        )
        
        # Article Details - Filter by selected topics
        doc_topics_info = []
        for doc, main_topic in zip([corpus[i] for i in mask], [doc_topic_mapping[i] for i in mask]):
            topic_dist = lda_model.get_document_topics(doc)
            topic_info = [
                f"Topic {topic+1}: {prob:.3f}"
                for topic, prob in sorted(topic_dist, key=lambda x: x[1], reverse=True)
                if topic in selected_topics  # Only include selected topics
            ]
            doc_topics_info.append(topic_info)
        
        articles_data = [{
            'title': row['title'],
            'section': row['section'],
            'published': row['published'].strftime('%Y-%m-%d %H:%M'),
            'topics': '\n'.join(topics)
        } for row, topics in zip(filtered_df.to_dict('records'), doc_topics_info)]
        
        logger.info("Successfully updated all visualizations")
        return dist_fig, word_cloud_fig, tsne_fig, articles_data
        
    except Exception as e:
        logger.error(f"Main callback error: {str(e)}", exc_info=True)
        empty_fig = go.Figure().update_layout(template='plotly_dark')
        return empty_fig, empty_fig, empty_fig, []

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
