##!/usr/bin/env python
# -*- coding: utf-8 -*-

import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
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

# ─────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Stop words
# ─────────────────────────────────────────────────────────────────────
CUSTOM_STOP_WORDS = {
    'says', 'said', 'would', 'also', 'one', 'new',
    'us', 'people', 'government', 'could', 'will',
    'may', 'like', 'get', 'make', 'first', 'two',
    'year', 'years', 'time', 'way', 'says', 'trump',
    'according', 'told', 'reuters', 'guardian',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'week', 'month'
}

# ─────────────────────────────────────────────────────────────────────
# Environment variables & NLTK
# ─────────────────────────────────────────────────────────────────────
load_dotenv()
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
if not GUARDIAN_API_KEY:
    logger.error("⚠️ No GUARDIAN_API_KEY found in environment!")
    raise ValueError("No GUARDIAN_API_KEY found in environment!")

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(CUSTOM_STOP_WORDS)

# ─────────────────────────────────────────────────────────────────────
# GuardianFetcher
# ─────────────────────────────────────────────────────────────────────
guardian = GuardianFetcher(GUARDIAN_API_KEY)

# ─────────────────────────────────────────────────────────────────────
# Dash Setup
# ─────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def process_articles(start_date, end_date):
    """
    Fetch articles from Guardian, filter by date, tokenize, train LDA.
    Returns (df, texts, dictionary, corpus, lda_model).
    """
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date}")
        
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        days_back = (datetime.now().date() - start_date_dt).days + 1
        
        df = guardian.fetch_articles(days_back=days_back, page_size=200)
        if df.empty:
            logger.warning("No articles fetched!")
            return None, None, None, None, None
        
        df = df[
            (df['published'].dt.date >= start_date_dt) &
            (df['published'].dt.date <= end_date_dt)
        ]
        logger.info(f"Filtered to {len(df)} articles in date range")
        if len(df) < 5:
            logger.warning("Not enough articles for LDA.")
            return None, None, None, None, None
        
        df.reset_index(drop=True, inplace=True)
        
        # Tokenize
        texts = []
        for i in range(len(df)):
            content = df.at[i, 'content']
            if pd.isna(content):
                texts.append([])
                continue
            words = word_tokenize(str(content))
            filtered = [
                w.lower() for w in words
                if w.isalnum() and w.lower() not in stop_words
            ]
            texts.append(filtered)
        
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(t) for t in texts]
        
        # Train LDA
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=5,
            id2word=dictionary,
            passes=10,
            random_state=42,
            chunksize=100
        )
        
        logger.info(f"Processed {len(df)} articles successfully")
        return df, texts, dictionary, corpus, lda_model
    except Exception as e:
        logger.error(f"Error in process_articles: {e}", exc_info=True)
        return None, None, None, None, None

# ─────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────
def create_word_cloud(topic_words):
    """
    Word cloud from LDA topic-word pairs, white background.
    """
    try:
        freq_dict = dict(topic_words)
        wc = WordCloud(
            background_color='white',
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
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly')

def create_tsne_visualization_3d(df, corpus, lda_model):
    """
    3D t-SNE scatter: n_components=3, scatter_3d.
    """
    try:
        if df is None or len(df) < 2:
            return go.Figure().update_layout(
                template='plotly',
                title='Not enough documents for t-SNE'
            )
        
        doc_topics_list = []
        for i in df.index:
            topic_weights = [0.0]*lda_model.num_topics
            for topic_id, w in lda_model[corpus[i]]:
                topic_weights[topic_id] = w
            doc_topics_list.append(topic_weights)
        
        doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
        if len(doc_topics_array) < 2:
            return go.Figure().update_layout(
                template='plotly',
                title='Not enough docs for t-SNE'
            )
        
        perplex_val = 30
        if len(doc_topics_array) < 30:
            perplex_val = max(2, len(doc_topics_array) - 1)
        
        tsne = TSNE(
            n_components=3,  # 3D
            random_state=42,
            perplexity=perplex_val,
            n_jobs=1
        )
        embedded = tsne.fit_transform(doc_topics_array)  # shape: (N, 3)
        
        scatter_df = pd.DataFrame({
            'x': embedded[:, 0],
            'y': embedded[:, 1],
            'z': embedded[:, 2],
            'dominant_topic': [np.argmax(row) for row in doc_topics_array],
            'doc_index': df.index,
            'title': df['title']
        })
        
        fig = px.scatter_3d(
            scatter_df,
            x='x', y='y', z='z',
            color='dominant_topic',
            hover_data=['title'],
            title='3D t-SNE Topic Clustering'
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating 3D t-SNE: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=f"t-SNE Error: {e}")

def create_bubble_chart(df):
    """
    Create a bubble chart showing doc length vs. published date, colored by top topic.
    We'll measure doc length simply as token count for each row.
    For color, we'll guess the doc's 'dominant_topic' from the df if available.
    
    If we haven't stored that, we can do a quick re-check (like get_document_topics).
    But for simplicity, let's do a naive approach:  we'll pick the first topic from the aggregated LDA model or store it earlier.
    We'll do that in the main callback.
    """
    try:
        if df is None or df.empty or 'doc_length' not in df.columns or 'dominant_topic' not in df.columns:
            return go.Figure().update_layout(
                template='plotly',
                title='Bubble Chart Unavailable'
            )
        
        fig = px.scatter(
            df,
            x='published',
            y='doc_length',
            size='doc_length',  # doc length as bubble size
            color='dominant_topic',  # color by dominant topic
            hover_data=['title'],
            title='Document Length Bubble Chart'
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=f"Bubble Chart Error: {e}")

# ─────────────────────────────────────────────────────────────────────
# Theming (Guardian-like)
# ─────────────────────────────────────────────────────────────────────
NAVY_BLUE = "#052962"

navbar = dbc.Navbar(
    [
        dbc.Row(
            [
                dbc.Col(html.Img(src="", height="30px"), width="auto"),
                dbc.Col(
                    dbc.NavbarBrand(
                        "Guardian News Topic Explorer",
                        className="ms-2",
                        style={"color": "white", "fontWeight": "bold", "fontSize": "1.2rem"}
                    )
                )
            ],
            align="center",
            className="g-0",
        )
    ],
    color=NAVY_BLUE,
    dark=True,
    className="mb-2 px-3"
)

controls_row = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Select Date Range", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
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
                            className="mb-3"
                        ),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=(datetime.now() - timedelta(days=30)).date(),
                            end_date=datetime.now().date(),
                            className="mb-2"
                        )
                    ]),
                ],
                className="mb-2",
                style={"backgroundColor": "white"}
            ),
            md=4
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Select Topics", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='topic-filter',
                            options=[{'label': f'Topic {i}', 'value': i} for i in range(5)],
                            multi=True,
                            placeholder="Filter by topics...",
                            className="mb-2"
                        )
                    ]),
                ],
                className="mb-2",
                style={"backgroundColor": "white"}
            ),
            md=4
        ),
        dbc.Col(md=4)  # Blank space or future expansions
    ],
    className="my-2 px-2"
)

# ─────────────────────────────────────────────────────────────────────
# Cards for the stacked layout
# 1) Topic Word Dist (top)
# 2) 3D t-SNE (middle)
# 3) Word Cloud (below)
# 4) Bubble Chart (below word cloud)
# 5) Articles Table
# ─────────────────────────────────────────────────────────────────────

topic_dist_card = dbc.Card(
    [
        dbc.CardHeader("Topic Word Distributions", style={"backgroundColor": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Graph(id='topic-distribution', style={"height": "600px"}),
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

tsne_3d_card = dbc.Card(
    [
        dbc.CardHeader("3D t-SNE Topic Clustering", style={"backgroundColor": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Graph(id='tsne-plot', style={"height": "600px"}),
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

wordcloud_card = dbc.Card(
    [
        dbc.CardHeader("Word Cloud", style={"backgroundColor": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Graph(id='word-cloud', style={"height": "600px"}),
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

bubble_chart_card = dbc.Card(
    [
        dbc.CardHeader("Document Length Bubble Chart", style={"backgroundColor": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Graph(id='bubble-chart', style={"height": "600px"}),
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

article_table_card = dbc.Card(
    [
        dbc.CardHeader("Article Details", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
        dbc.CardBody([
            dash_table.DataTable(
                id='article-details',
                columns=[
                    {'name': 'Title', 'id': 'title'},
                    {'name': 'Section', 'id': 'section'},
                    {'name': 'Published', 'id': 'published'},
                    {'name': 'Topics', 'id': 'topics'},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': 'white',
                    'color': 'black',
                    'textAlign': 'left',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_header={
                    'backgroundColor': NAVY_BLUE,
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                page_size=10
            )
        ], style={"backgroundColor": "white"})
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

app.layout = dbc.Container([
    navbar,
    controls_row,
    dbc.Row([dbc.Col(topic_dist_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(tsne_3d_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(wordcloud_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(bubble_chart_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(article_table_card, md=12)], className="g-3"),
], fluid=True, style={"backgroundColor": "#f9f9f9"})

# ─────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────

@app.callback(
    [Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    Input('date-select-buttons', 'value')
)
def update_date_range(selected_range):
    """
    Sync date picker with radio items.
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
        Output('bubble-chart', 'figure'),
        Output('article-details', 'data')
    ],
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('topic-filter', 'value')
    ]
)
def update_visuals(start_date, end_date, selected_topics):
    """
    Main callback:
    1) Topic Word Distributions
    2) Word Cloud
    3) 3D t-SNE
    4) Bubble Chart
    5) Article Table
    """
    try:
        logger.info(f"update_visuals: {start_date} to {end_date}, topics={selected_topics}")
        
        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date)
        if df is None or df.empty:
            empty_fig = go.Figure().update_layout(template='plotly', title="No Data")
            return empty_fig, empty_fig, empty_fig, empty_fig, []
        
        # If no topic selected, default to all topics [0..4]
        if not selected_topics:
            selected_topics = list(range(lda_model.num_topics))
        
        # 1) Topic Word Distributions
        words_list = []
        for t_id in selected_topics:
            top_pairs = lda_model.show_topic(t_id, topn=20)
            for (w, prob) in top_pairs:
                words_list.append((w, prob, t_id))
        if not words_list:
            dist_fig = go.Figure().update_layout(template='plotly', title="No topics selected")
        else:
            df_dist = pd.DataFrame(words_list, columns=["word", "prob", "topic"])
            dist_fig = px.bar(
                df_dist,
                x="prob",
                y="word",
                color="topic",
                orientation="h",
                title="Topic Word Distributions"
            )
            dist_fig.update_layout(template='plotly', yaxis={'categoryorder': 'total ascending'})
        
        # 2) Word Cloud (take first selected topic)
        first_topic = selected_topics[0]
        wc_fig = create_word_cloud(lda_model.show_topic(first_topic, topn=30))
        
        # 3) 3D t-SNE
        tsne_fig = create_tsne_visualization_3d(df, corpus, lda_model)
        
        # 4) Bubble Chart
        # We'll compute doc length for each doc, plus a "dominant_topic"
        # that is the highest weighted topic for that doc. 
        doc_lengths = []
        doc_dominant_topics = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            # doc length is just the sum of token counts
            n_tokens = len(texts[i]) if texts[i] else 0
            doc_lengths.append(n_tokens)
            # pick dominant topic
            if doc_topics:
                best_t = max(doc_topics, key=lambda x: x[1])[0]
            else:
                best_t = -1
            doc_dominant_topics.append(best_t)
        
        df["doc_length"] = doc_lengths
        df["dominant_topic"] = doc_dominant_topics
        
        # We'll do a bubble chart with x=published, y=doc_length, size=doc_length, color=dominant_topic
        bubble_fig = create_bubble_chart(df)
        
        # 5) Table data
        table_data = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            these_topics = []
            for (tid, w) in sorted(doc_topics, key=lambda x: x[1], reverse=True):
                if tid in selected_topics:
                    these_topics.append(f"Topic {tid}: {w:.3f}")
            table_data.append({
                'title': df.at[i, 'title'],
                'section': df.at[i, 'section'],
                'published': df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(these_topics)
            })
        
        return dist_fig, wc_fig, tsne_fig, bubble_fig, table_data
    
    except Exception as e:
        logger.error(f"update_visuals error: {e}", exc_info=True)
        empty_fig = go.Figure().update_layout(template='plotly', title=f"Error: {e}")
        return empty_fig, empty_fig, empty_fig, empty_fig, []

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
