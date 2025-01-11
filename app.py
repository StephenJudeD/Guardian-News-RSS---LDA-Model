#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Dash app showcasing:
1. Larger, more prominent Word Cloud
2. Topic trend over time (stacked area chart)
3. Click-to-filter from t-SNE scatter to show relevant articles
4. Snippet summaries for each article
5. Tabbed layout for a neat, multi-section interface
6. "About" tab providing extra context

Dependencies (from requirements.txt):
-------------------------------------
dash==2.9.3
dash-bootstrap-components==1.4.1
plotly==5.13.1
pandas==1.5.3
numpy==1.24.2
gensim==4.3.2
scipy==1.11.4
nltk==3.8.1
requests==2.28.2
gunicorn==20.1.0
python-dotenv==1.0.0
scikit-learn==1.3.2
wordcloud==1.9.2

Assumes you have a guardian_fetcher.py that provides:
    class GuardianFetcher:
        def __init__(self, api_key: str):
            ...
        def fetch_articles(self, days_back=30, page_size=200):
            ...
            return df  # a DataFrame with columns: ['title', 'section', 'content', 'published']
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

# Enhanced stop words / custom stopwords
CUSTOM_STOP_WORDS = {
    'says', 'said', 'would', 'also', 'one', 'new',
    'us', 'people', 'government', 'could', 'will',
    'may', 'like', 'get', 'make', 'first', 'two',
    'year', 'years', 'time', 'way', 'says', 'trump',
    'according', 'told', 'reuters', 'guardian',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'week', 'month'
}

load_dotenv()
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
if not GUARDIAN_API_KEY:
    logger.error("⚠️ No GUARDIAN_API_KEY found in environment!")
    raise ValueError("No GUARDIAN_API_KEY found in environment!")

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(CUSTOM_STOP_WORDS)

# Initialize our GuardianFetcher
guardian = GuardianFetcher(GUARDIAN_API_KEY)

# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])
server = app.server
app.config.suppress_callback_exceptions = True

@lru_cache(maxsize=64)
def process_articles(start_date, end_date):
    """
    Fetch and process Guardian articles into
    a DataFrame, plus the LDA model resources.
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
        
        # Minimal content processing: tokenization + stopwords removal
        texts = []
        for i in df.index:
            content = df.at[i, 'content']
            if pd.isna(content):
                # Just store empty if no content
                texts.append([])
                continue
            words = word_tokenize(str(content))
            filtered_words = [
                word.lower() for word in words
                if word.isalnum() and word.lower() not in stop_words
            ]
            texts.append(filtered_words)
        
        # If too few articles, skip
        if len(texts) < 5:
            logger.warning("Not enough articles for analysis!")
            return None, None, None, None, None
        
        # Summaries: store first ~120 chars of each for a quick snippet
        df['snippet'] = df['content'].apply(
            lambda x: (str(x)[:120] + '...') if isinstance(x, str) else ''
        )
        
        # Create dictionary, corpus, train LDA
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=5,
            id2word=dictionary,
            passes=10,
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
    Creates a larger, more eye-catching word cloud
    from a list of (word, probability) pairs.
    """
    try:
        freq_dict = dict(topic_words)
        wc = WordCloud(
            background_color='black',
            width=1200,  # made bigger
            height=600,  # made bigger
            colormap='viridis'
        ).generate_from_frequencies(freq_dict)
        fig = px.imshow(wc)
        fig.update_layout(
            template='plotly',
            title="Topic Word Cloud (Enlarged)"
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {str(e)}", exc_info=True)
        return go.Figure().update_layout(template='plotly')

def create_tsne_visualization(corpus, lda_model, df):
    """
    Creates an interactive t-SNE plot of topics.
    We add a custom_data attribute to let user
    click on points and filter the table.
    """
    try:
        doc_topics = []
        for doc in corpus:
            topic_weights = [0]*lda_model.num_topics
            for t_id, weight in lda_model[doc]:
                topic_weights[t_id] = weight
            doc_topics.append(topic_weights)
        
        tsne = TSNE(n_components=2, random_state=42)
        topics_2d = tsne.fit_transform(doc_topics)
        
        df_viz = pd.DataFrame({
            'x': topics_2d[:, 0],
            'y': topics_2d[:, 1],
            'dominant_topic': [np.argmax(doc_topics[i])+1 for i in range(len(doc_topics))],
            # storing the original index so we can filter the DataTable
            'doc_index': df.index,
            'title': df['title']
        })
        
        fig = px.scatter(
            df_viz,
            x='x', y='y',
            color='dominant_topic',
            hover_data=['title'],
            title='t-SNE Topic Clustering',
            custom_data=['doc_index']  # important for click event
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating t-SNE: {str(e)}", exc_info=True)
        return go.Figure().update_layout(template='plotly')

def calculate_topic_trend(df, corpus, lda_model):
    """
    Computes how many articles belong to each topic per day,
    returning a DataFrame suitable for a stacked area chart.
    """
    # For each doc, find the dominant topic
    dominant_topics = []
    for doc in corpus:
        topic_weights = [0]*lda_model.num_topics
        for t_id, w in lda_model[doc]:
            topic_weights[t_id] = w
        dom_topic = np.argmax(topic_weights)
        dominant_topics.append(dom_topic)
    
    df_temp = df.copy()
    df_temp['dom_topic'] = dominant_topics
    df_temp['pub_date'] = df_temp['published'].dt.date
    
    # Group by date & topic
    group = df_temp.groupby(['pub_date', 'dom_topic']).size().reset_index(name='count')
    
    # pivot so each topic is a separate column
    pivoted = group.pivot(index='pub_date', columns='dom_topic', values='count').fillna(0)
    pivoted.sort_index(inplace=True)
    # rename columns so they become 'Topic 1', 'Topic 2'...
    pivoted.columns = [f'Topic {c+1}' for c in pivoted.columns]
    pivoted.index = pd.to_datetime(pivoted.index)  # Ensure DateTime index
    return pivoted

def create_topic_trend_figure(df_trend):
    """
    Creates a stacked area chart from the pivoted time-indexed DF.
    """
    if df_trend.empty:
        return go.Figure().update_layout(template='plotly', title="No data for trend")
    
    fig = go.Figure()
    for col in df_trend.columns:
        fig.add_trace(
            go.Scatter(
                x=df_trend.index, 
                y=df_trend[col],
                name=col,
                stackgroup='one',
                mode='lines',
            )
        )
    fig.update_layout(
        template='plotly',
        title='Topic Trend Over Time (Stacked Area)',
        xaxis_title='Date',
        yaxis_title='Article Count'
    )
    return fig

# ----------------------------------------------------------------------------
# LAYOUT
# ----------------------------------------------------------------------------

# Navbar
navbar = dbc.NavbarSimple(
    brand="Guardian News Topic Explorer",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# A banner that replaces the old Jumbotron
banner = dbc.Container(
    [
        html.H1("Guardian News Topic Explorer 📰", className="display-3 fw-bold"),
        html.P(
            "Interactive topic modeling, t-SNE clustering, and more.",
            className="lead text-muted"
        )
    ],
    fluid=True,
    className="py-4 my-4 bg-light rounded-3 text-center"
)

# Controls (left side)
controls_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Controls")),
        dbc.CardBody(
            [
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
            ]
        )
    ],
    className="mb-4 shadow"
)

# Graph cards
topic_distribution_card = dbc.Card(
    [
        dbc.CardHeader("Topic Word Distributions", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='topic-distribution', style={"height": "400px"})
        ])
    ],
    className="mb-4 shadow"
)

word_cloud_card = dbc.Card(
    [
        dbc.CardHeader("Word Cloud", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='word-cloud', style={"height": "600px"})  # made bigger
        ])
    ],
    className="mb-4 shadow"
)

tsne_plot_card = dbc.Card(
    [
        dbc.CardHeader("t-SNE Topic Clustering (Click a point)", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='tsne-plot', style={"height": "600px"})
        ])
    ],
    className="mb-4 shadow"
)

topic_trend_card = dbc.Card(
    [
        dbc.CardHeader("Topic Trend Over Time", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='topic-trend', style={"height": "400px"})
        ])
    ],
    className="mb-4 shadow"
)

articles_table_card = dbc.Card(
    [
        dbc.CardHeader("Article Details", className="bg-secondary text-light"),
        dbc.CardBody([
            dash_table.DataTable(
                id='article-details',
                columns=[
                    {'name': 'Title', 'id': 'title'},
                    {'name': 'Section', 'id': 'section'},
                    {'name': 'Published', 'id': 'published'},
                    {'name': 'Topics', 'id': 'topics'},
                    {'name': 'Snippet', 'id': 'snippet'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
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

# Tab 1 (Main Dashboard)
tab_main_content = dbc.Container(
    [
        banner,
        dbc.Row([
            dbc.Col(controls_card, md=4),
            dbc.Col([
                topic_distribution_card,
                word_cloud_card
            ], md=8)
        ]),
        dbc.Row([
            dbc.Col(tsne_plot_card, md=12)
        ]),
        dbc.Row([
            dbc.Col(topic_trend_card, md=12)
        ]),
        dbc.Row([
            dbc.Col(articles_table_card, md=12)
        ])
    ],
    fluid=True
)

# Tab 2 (About / How It Works)
tab_about_content = dbc.Container(
    [
        html.H2("About This Dashboard", className="mt-4"),
        html.P(
            """
            This app fetches recent Guardian articles and applies techniques such as:
            1) LDA topic modeling
            2) t-SNE for clustering
            3) WordCloud for a quick visual display of top words
            4) Time-series analysis to see how certain topics fluctuate over time
            """
        ),
        html.P(
            """
            Interactivity: 
            • Click the t-SNE scatter to filter articles. 
            • Select date range or specific topics to refine the analysis.
            This is intended as a data science demonstration to show how
            text analytics can help uncover hidden structures in news articles.
            """
        )
    ],
    fluid=True
)

# The main layout with dcc.Tabs
app.layout = dbc.Container([
    navbar,
    dcc.Tabs([
        dcc.Tab(label='Dashboard', children=[tab_main_content]),
        dcc.Tab(label='About', children=[tab_about_content])
    ], style={"marginTop": "20px"})
], fluid=True)

# ----------------------------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------------------------

@app.callback(
    [Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    [Input('date-select-buttons', 'value')]
)
def update_date_range(selected_range):
    """
    Keep the date range picker in sync with preset radio items
    (Last Day, Last Week, Last Month).
    """
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
        Output('topic-trend', 'figure'),
        Output('article-details', 'data')
    ],
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('topic-filter', 'value'),
        Input('tsne-plot', 'clickData')  # for click to filter
    ]
)
def update_visualizations(start_date, end_date, selected_topics, tsne_click):
    """
    Main callback that:
    1. Fetches/processes articles
    2. Builds topic distribution bar chart
    3. Builds a bigger word cloud
    4. Builds t-SNE scatter (with custom_data to track doc_index)
    5. Builds topic trend stacked area chart
    6. Displays filtered articles in a table
    7. Allows user to click t-SNE points to further filter the data
    """
    try:
        logger.info(f"Updating visuals from {start_date} to {end_date}, topics={selected_topics}")
        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date)
        if df is None or df.empty:
            # Return empty or placeholders
            return [go.Figure(), go.Figure(), go.Figure(), go.Figure(), []]
        
        # Default: show all topics if none selected
        num_topics = lda_model.num_topics
        if not selected_topics:
            selected_topics = list(range(num_topics))
        
        # 1) Topic Distribution
        #   gather top words across selected topics
        topic_terms = []
        for topic_id in selected_topics:
            for word, prob in lda_model.show_topic(topic_id, 10):
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
        
        # 2) Word Cloud for first selected topic
        #    (or zero if none)
        sel_topic = selected_topics[0] if selected_topics else 0
        wc_fig = create_word_cloud(lda_model.show_topic(sel_topic, topn=30))
        
        # 3) t-SNE scatter
        tsne_fig = create_tsne_visualization(corpus, lda_model, df)
        
        # 4) Topic Trend Over Time
        df_trend = calculate_topic_trend(df, corpus, lda_model)
        trend_fig = create_topic_trend_figure(df_trend)
        
        # 5) Filter articles
        #   a) by selected_topics
        #   b) by clicked t-SNE point (doc_index)
        doc_topics_list = []
        doc_topic_mapping = []
        for i, doc in enumerate(corpus):
            topic_weights = [0]*num_topics
            for t_id, w in lda_model[doc]:
                topic_weights[t_id] = w
            doc_topics_list.append(topic_weights)
            dom_topic = np.argmax(topic_weights)
            doc_topic_mapping.append(dom_topic)
        
        # build mask
        mask = [True]*len(df)  # start with including all
        # filter by selected topics
        for idx in range(len(df)):
            if doc_topic_mapping[idx] not in selected_topics:
                mask[idx] = False
        
        # if user clicked t-SNE
        clicked_index = None
        if tsne_click and 'points' in tsne_click:
            # each point has customdata = [doc_index]
            clicked_index = tsne_click['points'][0]['customdata'][0]
            # filter further so we only keep that doc
            # or (maybe you want keep doc from same cluster,
            # or however you want to handle that).
            # For example, let's do only that doc:
            new_mask = [False]*len(df)
            if 0 <= clicked_index < len(df):
                new_mask[clicked_index] = True
            # combine with existing mask
            mask = [m and nm for m, nm in zip(mask, new_mask)]
        
        final_df = df[mask].copy()
        
        # Build Table data
        articles_data = []
        for i in final_df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            # keep only topics in selected_topics
            topic_str_list = []
            for t_id, prob in sorted(doc_topics, key=lambda x: x[1], reverse=True):
                if t_id in selected_topics:
                    topic_str_list.append(f"Topic {t_id+1}: {prob:.3f}")
            
            articles_data.append({
                'title': final_df.at[i, 'title'],
                'section': final_df.at[i, 'section'],
                'published': final_df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(topic_str_list),
                'snippet': final_df.at[i, 'snippet']
            })
        
        return dist_fig, wc_fig, tsne_fig, trend_fig, articles_data
    
    except Exception as e:
        logger.error(f"Main callback error: {str(e)}", exc_info=True)
        # Return empties
        empty_fig = go.Figure().update_layout(template='plotly')
        return empty_fig, empty_fig, empty_fig, empty_fig, []

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
