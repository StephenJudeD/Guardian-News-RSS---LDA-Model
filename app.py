#!/usr/bin/env python3

import dash
from dash import Dash, html, dcc, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from guardian_fetcher import GuardianFetcher
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser

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
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Expanded Stop Words
# ─────────────────────────────────────────────────────────────────────
CUSTOM_STOP_WORDS = {
    'says', 'said', 'would', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'new', 'like', 'get', 'make', 'first', 'year', 'years', 'time', 'way', 'says', 'say', 'saying', 'according',
    'told', 'reuters', 'guardian', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'week', 'month', 'us', 'people', 'government', 'could', 'will', 'may', 'trump', 'published', 'article', 'editor',
    'nt', 'dont', 'doesnt', 'cant', 'couldnt', 'shouldnt', 'last', 'well', 'still', 'price',
    # Added more for demonstration:
    'breaking', 'update', 'live'
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
# Default is dark theme, user can toggle to light theme
external_stylesheets = [dbc.themes.DARKLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

THEME_URLS = {
    "dark": dbc.themes.DARKLY,
    "light": dbc.themes.BOOTSTRAP
}

# ─────────────────────────────────────────────────────────────────────
# Dark/Light Plotly Layout Helpers
# ─────────────────────────────────────────────────────────────────────
def get_plotly_dark_layout(fig_title=""):
    """Return a default layout for dark-mode figures with grid lines, reduced margins."""
    return dict(
        paper_bgcolor="#303030",
        plot_bgcolor="#303030",
        font_color="white",
        title=fig_title,
        margin=dict(l=20, r=20, t=60, b=40),  # NEW/CHANGED: reduce margins
    )

def get_plotly_light_layout(fig_title=""):
    """Return a default layout for light-mode figures with grid lines, reduced margins."""
    return dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        title=fig_title,
        margin=dict(l=20, r=20, t=60, b=40),  # NEW/CHANGED: reduce margins
    )

# ─────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def process_articles(start_date, end_date, num_topics=5):
    """
    Fetch Guardian articles in the given date range,
    then tokenize, detect bigrams/trigrams, and train LDA on the entire set.
    Returns (df, texts, dictionary, corpus, lda_model).
    """
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date} with num_topics={num_topics}")

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
        tokenized_texts = []
        for i in df.index:
            content = df.at[i, 'content']
            if pd.isna(content):
                tokenized_texts.append([])
                continue
            words = word_tokenize(str(content))
            filtered = [
                w.lower() for w in words
                if w.isalnum() and w.lower() not in stop_words
            ]
            tokenized_texts.append(filtered)

        # Bigrams & Trigrams
        bigram_phrases = Phrases(tokenized_texts, min_count=5, threshold=10)
        trigram_phrases = Phrases(bigram_phrases[tokenized_texts], threshold=10)
        bigram = Phraser(bigram_phrases)
        trigram = Phraser(trigram_phrases)

        texts = []
        for t in tokenized_texts:
            bigrammed = bigram[t]
            trigrammed = trigram[bigrammed]
            texts.append(trigrammed)

        # Dictionary & Corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(t) for t in texts]

        # Train LDA with dynamic num_topics
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=10,
            random_state=42,
            chunksize=100
        )

        logger.info(f"Processed {len(df)} articles successfully with LDA num_topics={num_topics}")
        return df, texts, dictionary, corpus, lda_model

    except Exception as e:
        logger.error(f"Error in process_articles: {e}", exc_info=True)
        return None, None, None, None, None

# ─────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────

def setup_axes_grid(fig, dark_mode=False):
    """ # NEW/CHANGED: Helper function to enable grid lines on x/y axes. """
    grid_color = "#666" if dark_mode else "#ccc"
    fig.update_xaxes(showgrid=True, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color)

def create_word_cloud(topic_words, dark_mode=False):
    """
    Create a word cloud from LDA topic-word pairs.
    If dark_mode=True, word cloud background is black, else white.
    """
    try:
        freq_dict = dict(topic_words)
        bg = "black" if dark_mode else "white"

        # NEW/CHANGED: Increase size for a bigger word cloud
        wc = WordCloud(
            background_color=bg,
            width=1000,   # bigger width
            height=550,   # bigger height
            colormap='viridis'
        ).generate_from_frequencies(freq_dict)

        fig = px.imshow(wc)

        if dark_mode:
            fig.update_layout(**get_plotly_dark_layout("Topic Word Cloud"))
        else:
            fig.update_layout(**get_plotly_light_layout("Topic Word Cloud"))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        return go.Figure()

def create_tsne_visualization_3d(df, corpus, lda_model, perplexity=30, dark_mode=False):
    """
    3D t-SNE scatter (Plotly).
    Uses all documents in df/corpus to avoid filtering by topic.
    Allows dynamic perplexity setting.
    """
    try:
        if df is None or len(df) < 2:
            fig = go.Figure()
            if dark_mode:
                fig.update_layout(**get_plotly_dark_layout("Not enough documents for t-SNE"))
            else:
                fig.update_layout(**get_plotly_light_layout("Not enough documents for t-SNE"))
            return fig

        doc_topics_list = []
        for i in df.index:
            topic_weights = [0.0]*lda_model.num_topics
            for topic_id, w in lda_model[corpus[i]]:
                topic_weights[topic_id] = w
            doc_topics_list.append(topic_weights)

        doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
        if len(doc_topics_array) < 2:
            fig = go.Figure()
            if dark_mode:
                fig.update_layout(**get_plotly_dark_layout("Not enough docs for t-SNE"))
            else:
                fig.update_layout(**get_plotly_light_layout("Not enough docs for t-SNE"))
            return fig

        perplex_val = min(perplexity, max(2, len(doc_topics_array) - 1))
        tsne = TSNE(
            n_components=3,
            random_state=42,
            perplexity=perplex_val,
            n_jobs=1
        )
        embedded = tsne.fit_transform(doc_topics_array)

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
            title=f'3D t-SNE Topic Clustering (Perplexity={perplex_val})'
        )
        if dark_mode:
            fig.update_layout(**get_plotly_dark_layout())
        else:
            fig.update_layout(**get_plotly_light_layout())

        # NEW/CHANGED: add grid lines
        setup_axes_grid(fig, dark_mode=dark_mode)

        return fig
    except Exception as e:
        logger.error(f"Error creating 3D t-SNE: {e}", exc_info=True)
        fig = go.Figure()
        if dark_mode:
            fig.update_layout(**get_plotly_dark_layout(str(e)))
        else:
            fig.update_layout(**get_plotly_light_layout(str(e)))
        return fig

def create_bubble_chart(df, dark_mode=False):
    """
    Bubble chart: doc length vs published date, sized by doc length,
    colored by dominant_topic. Removes outliers & uses log scale.
    """
    try:
        if df is None or df.empty:
            fig = go.Figure()
            if dark_mode:
                fig.update_layout(**get_plotly_dark_layout("Bubble Chart Unavailable"))
            else:
                fig.update_layout(**get_plotly_light_layout("Bubble Chart Unavailable"))
            return fig

        cut_off = df['doc_length'].quantile(0.95)
        filtered_df = df[df['doc_length'] <= cut_off].copy()
        if filtered_df.empty:
            fig = go.Figure()
            if dark_mode:
                fig.update_layout(**get_plotly_dark_layout("No Data after outlier removal"))
            else:
                fig.update_layout(**get_plotly_light_layout("No Data after outlier removal"))
            return fig

        fig = px.scatter(
            filtered_df,
            x='published',
            y='doc_length',
            size='doc_length',
            color='dominant_topic',
            size_max=30,
            hover_data=['title'],
            title='Document Length Bubble Chart (w/ Outlier Removal & Log Scale)',
            log_y=True
        )
        if dark_mode:
            fig.update_layout(**get_plotly_dark_layout())
        else:
            fig.update_layout(**get_plotly_light_layout())

        # NEW/CHANGED: enable grid lines
        setup_axes_grid(fig, dark_mode=dark_mode)

        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        fig = go.Figure()
        if dark_mode:
            fig.update_layout(**get_plotly_dark_layout(str(e)))
        else:
            fig.update_layout(**get_plotly_light_layout(str(e)))
        return fig

def create_ngram_radar_chart(texts, dark_mode=False):
    """
    Radar (sonar) chart of the most common bigrams/trigrams (top 10).
    """
    try:
        ngram_counts = {}
        for tokens in texts:
            for tok in tokens:
                if "_" in tok:
                    ngram_counts[tok] = ngram_counts.get(tok, 0) + 1

        if not ngram_counts:
            fig = go.Figure()
            if dark_mode:
                fig.update_layout(**get_plotly_dark_layout("No bigrams/trigrams found"))
            else:
                fig.update_layout(**get_plotly_light_layout("No bigrams/trigrams found"))
            return fig

        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        top_ngrams = sorted_ngrams[:10]
        df_ngram = pd.DataFrame(top_ngrams, columns=["ngram", "count"])

        fig = px.line_polar(
            df_ngram,
            r="count",
            theta="ngram",
            line_close=True,
            title="Top Bigrams & Trigrams (Radar Chart)"
        )
        fig.update_traces(fill='toself')

        if dark_mode:
            fig.update_layout(**get_plotly_dark_layout())
        else:
            fig.update_layout(**get_plotly_light_layout())

        # Radar charts don’t show grid lines in the same way, but you can experiment with radial axes
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram radar chart: {e}", exc_info=True)
        fig = go.Figure()
        if dark_mode:
            fig.update_layout(**get_plotly_dark_layout(str(e)))
        else:
            fig.update_layout(**get_plotly_light_layout(str(e)))
        return fig

# ─────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────

# Theme toggle
theme_toggle = dbc.Checklist(
    options=[{"label": "Dark Mode", "value": 1}],
    value=[1],  # default is dark mode
    id="theme-toggle",
    switch=True,
    style={"marginLeft": "15px"}
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("Guardian News Topic Explorer", id="app-title", className="ms-2"),
                        width="auto",
                    ),
                ],
                align="center",
                className="g-0",
            ),
            dbc.Row(
                [
                    dbc.Col(theme_toggle, width="auto"),
                ],
                align="center",
            )
        ],
        fluid=True
    ),
    id="navbar",
    color="dark",
    dark=True,
    className="mb-2 px-3"
)

explainer_card = dbc.Card(
    [
        dbc.CardHeader("About This App", id="about-header"),
        dbc.CardBody(
            [
                html.P(
                    [
                        "This dashboard fetches articles from the Guardian’s RSS, processes them with "
                        "Natural Language Processing (NLP), and then applies techniques like LDA for topic modeling, "
                        "bigrams/trigrams detection for multi-word phrases, and t-SNE for visualizing clusters in 3D. "
                        "Use the controls below to explore the data: date range, dynamic LDA topic counts, perplexity, "
                        "and more. See how news stories shift over time! ",
                        html.A(
                            "Code & Readme available @ GitHub",
                            href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model/tree/main",
                            target="_blank",
                            style={"textDecoration": "underline"},
                            id="github-link"
                        ),
                    ],
                    className="mb-0",
                    id="about-text"
                )
            ],
            id="about-body"
        ),
    ],
    className="mb-3",
    id="about-card"
)

date_filter_card = dbc.Card(
    [
        dbc.CardHeader("Select Date Range", id="date-header"),
        dbc.CardBody([
            dbc.RadioItems(
                id='date-select-buttons',
                options=[
                    {'label': 'Last Day', 'value': 'last_day'},
                    {'label': 'Last 3 Days', 'value': 'last_three'},
                    {'label': 'Last Week', 'value': 'last_week'},
                ],
                value='last_week',
                inline=True,
                className="mb-3"
                # (Removed duplicate id to avoid error)
            ),
            dcc.DatePickerRange(
                id='date-range',
                start_date=(datetime.now() - timedelta(days=7)).date(),
                end_date=datetime.now().date()
            )
        ], id="date-body"),
    ],
    className="mb-2",
    id="date-card"
)

num_topics_card = dbc.Card(
    [
        dbc.CardHeader("LDA: Number of Topics", id="num-topics-header"),
        dbc.CardBody([
            dcc.Slider(
                id="num-topics-slider",
                min=2,
                max=15,
                value=5,
                step=1,
                marks={i: str(i) for i in range(2, 16)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="mb-2"
            ),
        ], id="num-topics-body"),
    ],
    className="mb-2",
    id="num-topics-card"
)

tsne_controls_card = dbc.Card(
    [
        dbc.CardHeader("t-SNE Perplexity", id="tsne-header"),
        dbc.CardBody([
            dcc.Slider(
                id='tsne-perplexity-slider',
                min=5,
                max=50,
                step=5,
                value=30,
                marks={i: str(i) for i in range(5, 51, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], id="tsne-body"),
    ],
    className="mb-2",
    id="tsne-card"
)

controls_row = dbc.Row(
    [
        dbc.Col(date_filter_card, md=4),
        dbc.Col(num_topics_card, md=4),
        dbc.Col(tsne_controls_card, md=4),
    ],
    className="my-2 px-2"
)

topic_dist_card = dbc.Card(
    [
        dbc.CardHeader("Topic Word Distributions (Top 10)", id="topic-dist-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-topic-dist",
                type="circle",
                children=[dcc.Graph(id='topic-distribution', style={"height": "600px"})]
            ),
            id="topic-dist-body"
        )
    ],
    className="mb-3",
    id="topic-dist-card"
)

tsne_3d_card = dbc.Card(
    [
        dbc.CardHeader("3D t-SNE Topic Clustering", id="tsne-3d-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-3d-tsne",
                type="circle",
                children=[dcc.Graph(id='tsne-plot', style={"height": "600px"})]
            ),
            id="tsne-3d-body"
        )
    ],
    className="mb-3",
    id="tsne-3d-card"
)

wordcloud_card = dbc.Card(
    [
        dbc.CardHeader("Topic Word Cloud", id="wordcloud-header"),  # NEW/CHANGED: clearer label
        dbc.CardBody(
            dcc.Loading(
                id="loading-wordcloud",
                type="circle",
                children=[dcc.Graph(id='word-cloud', style={"height": "600px"})]
            ),
            id="wordcloud-body"
        )
    ],
    className="mb-3",
    id="wordcloud-card"
)

bubble_chart_card = dbc.Card(
    [
        dbc.CardHeader("Document Length Bubble Chart", id="bubble-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-bubble-chart",
                type="circle",
                children=[dcc.Graph(id='bubble-chart', style={"height": "600px"})]
            ),
            id="bubble-body"
        )
    ],
    className="mb-3",
    id="bubble-card"
)

bigrams_trigrams_card = dbc.Card(
    [
        dbc.CardHeader("Bigrams & Trigrams (Radar Chart)", id="bigrams-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-bigrams-trigrams",
                type="circle",
                children=[dcc.Graph(id='bigrams-trigrams', style={"height": "600px"})]
            ),
            id="bigrams-body"
        )
    ],
    className="mb-3",
    id="bigrams-card"
)

article_table_card = dbc.Card(
    [
        dbc.CardHeader("Article Details", id="articles-header"),
        dbc.CardBody([
            dash_table.DataTable(
                id='article-details',
                columns=[
                    {'name': 'Title', 'id': 'title'},
                    {'name': 'Published', 'id': 'published'},
                    {'name': 'Topics', 'id': 'topics'},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
                style_header={'fontWeight': 'bold'},
                page_size=10
            )
        ], id="articles-body")
    ],
    className="mb-3",
    id="articles-card"
)

app.layout = dbc.Container(
    [
        navbar,
        dbc.Row([dbc.Col(explainer_card, md=12)], className="g-3"),
        controls_row,
        dbc.Row([dbc.Col(topic_dist_card, md=12)], className="g-3"),
        dbc.Row([dbc.Col(tsne_3d_card, md=12)], className="g-3"),
        dbc.Row([dbc.Col(wordcloud_card, md=12)], className="g-3"),
        dbc.Row([dbc.Col(bubble_chart_card, md=12)], className="g-3"),
        dbc.Row([dbc.Col(bigrams_trigrams_card, md=12)], className="g-3"),
        dbc.Row([dbc.Col(article_table_card, md=12)], className="g-3"),
    ],
    fluid=True,
    id="main-container"
)

# ─────────────────────────────────────────────────────────────────────
# Theme Toggle Callback
# ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("main-container", "className"),
    Output("navbar", "color"),
    Output("navbar", "dark"),
    Output("app-title", "style"),
    Output("about-card", "style"),
    Output("github-link", "style"),
    [Output(c_id, "style") for c_id in [
        "about-header", "about-body", "topic-dist-header", "topic-dist-body",
        "tsne-3d-header", "tsne-3d-body", "wordcloud-header", "wordcloud-body",
        "bubble-header", "bubble-body", "bigrams-header", "bigrams-body",
        "articles-header", "articles-body", "date-header", "date-body",
        "num-topics-header", "num-topics-body", "tsne-header", "tsne-body"
    ]],
    Input("theme-toggle", "value")
)
def toggle_theme(theme_value):
    """
    If theme_value is [1], we use dark mode. Otherwise light mode.
    We'll manually style components to match the chosen mode.
    """
    is_dark = (theme_value == [1])
    navbar_color = "dark" if is_dark else "light"
    navbar_dark = True if is_dark else False

    title_style = {"color": "white"} if is_dark else {"color": "black"}
    card_bg = "#303030" if is_dark else "white"
    text_color = "white" if is_dark else "black"
    link_color = "lightblue" if is_dark else "blue"

    # Common style dict for card header/body
    style_for_header = {
        "backgroundColor": card_bg,
        "color": text_color,
        "fontWeight": "bold"
    }
    style_for_body = {
        "backgroundColor": card_bg,
        "color": text_color
    }
    style_for_link = {"color": link_color, "textDecoration": "underline"}

    return (
        ("dark-mode" if is_dark else ""),  # main-container className
        navbar_color,
        navbar_dark,
        title_style,
        {"backgroundColor": card_bg},  # about-card style
        style_for_link
    ) + tuple(
        style_for_header if "header" in c_id else style_for_body
        for c_id in [
            "about-header", "about-body", "topic-dist-header", "topic-dist-body",
            "tsne-3d-header", "tsne-3d-body", "wordcloud-header", "wordcloud-body",
            "bubble-header", "bubble-body", "bigrams-header", "bigrams-body",
            "articles-header", "articles-body", "date-header", "date-body",
            "num-topics-header", "num-topics-body", "tsne-header", "tsne-body"
        ]
    )

# ─────────────────────────────────────────────────────────────────────
# Date Range Callback
# ─────────────────────────────────────────────────────────────────────
@app.callback(
    [Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    Input('date-select-buttons', 'value')
)
def update_date_range(selected_range):
    end_date = datetime.now().date()
    if selected_range == 'last_day':
        start_date = end_date - timedelta(days=1)
    elif selected_range == 'last_three':
        start_date = end_date - timedelta(days=3)
    elif selected_range == 'last_week':
        start_date = end_date - timedelta(days=7)
    else:
        start_date = end_date - timedelta(days=7)
    return start_date, end_date

# ─────────────────────────────────────────────────────────────────────
# Main Visualization Callback
# ─────────────────────────────────────────────────────────────────────
@app.callback(
    [
        Output('topic-distribution', 'figure'),
        Output('word-cloud', 'figure'),
        Output('tsne-plot', 'figure'),
        Output('bubble-chart', 'figure'),
        Output('bigrams-trigrams', 'figure'),
        Output('article-details', 'data')
    ],
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('num-topics-slider', 'value'),
        Input('tsne-perplexity-slider', 'value'),
        Input('theme-toggle', 'value')
    ]
)
def update_visuals(start_date, end_date, num_topics, perplexity, theme_value):
    """
    1) Train LDA on the entire set within the selected date range.
    2) Build visuals & article table from the entire set.
    3) Create 3D t-SNE with user-chosen perplexity.
    4) Show bubble chart, etc.
    5) Dark/Light mode for figures.
    """
    try:
        logger.info(f"update_visuals: {start_date} to {end_date}, num_topics={num_topics}, perplexity={perplexity}")
        is_dark = (theme_value == [1])

        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date, num_topics)
        if df is None or df.empty:
            # Return empty figs
            fig_empty_dark = go.Figure().update_layout(**get_plotly_dark_layout("No Data"))
            fig_empty_light = go.Figure().update_layout(**get_plotly_light_layout("No Data"))
            fig_empty = fig_empty_dark if is_dark else fig_empty_light
            return fig_empty, fig_empty, fig_empty, fig_empty, fig_empty, []

        doc_lengths = []
        doc_dominant_topics = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            n_tokens = len(texts[i] if texts[i] else [])
            doc_lengths.append(n_tokens)
            if doc_topics:
                best_t = max(doc_topics, key=lambda x: x[1])[0]
            else:
                best_t = -1
            doc_dominant_topics.append(best_t)

        df["doc_length"] = doc_lengths
        df["dominant_topic"] = doc_dominant_topics

        # Topic Word Distribution
        words_list = []
        for t_id in range(num_topics):
            if 0 <= t_id < lda_model.num_topics:
                top_pairs = lda_model.show_topic(t_id, topn=10)
                for (w, prob) in top_pairs:
                    words_list.append((w, prob, t_id))

        if not words_list:
            fig_dist = go.Figure()
            if is_dark:
                fig_dist.update_layout(**get_plotly_dark_layout("No topics found"))
            else:
                fig_dist.update_layout(**get_plotly_light_layout("No topics found"))
        else:
            df_dist = pd.DataFrame(words_list, columns=["word", "prob", "topic"])
            fig_dist = px.bar(
                df_dist,
                x="prob",
                y="word",
                color="topic",
                orientation="h",
                title="Topic Word Distributions (Top 10)"
            )
            if is_dark:
                fig_dist.update_layout(**get_plotly_dark_layout())
            else:
                fig_dist.update_layout(**get_plotly_light_layout())

            # NEW/CHANGED: show grid lines, reduce empty space
            setup_axes_grid(fig_dist, dark_mode=is_dark)

        # Word Cloud (Topic 0)
        fig_wc = go.Figure()
        if num_topics > 0 and lda_model.num_topics > 0:
            fig_wc = create_word_cloud(lda_model.show_topic(0, topn=30), dark_mode=is_dark)
        else:
            if is_dark:
                fig_wc.update_layout(**get_plotly_dark_layout("Word Cloud N/A"))
            else:
                fig_wc.update_layout(**get_plotly_light_layout("Word Cloud N/A"))

        # 3D t-SNE
        fig_tsne = create_tsne_visualization_3d(df, corpus, lda_model, perplexity, dark_mode=is_dark)

        # Bubble Chart
        fig_bubble = create_bubble_chart(df, dark_mode=is_dark)

        # Radar Chart for Bigrams & Trigrams
        fig_ngram = create_ngram_radar_chart(texts, dark_mode=is_dark)

        # Article Table
        table_data = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            these_topics = [
                f"Topic {tid}: {w:.3f}" for (tid, w)
                in sorted(doc_topics, key=lambda x: x[1], reverse=True)
            ]
            table_data.append({
                'title': df.at[i, 'title'],
                'published': df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(these_topics)
            })

        return fig_dist, fig_wc, fig_tsne, fig_bubble, fig_ngram, table_data

    except Exception as e:
        logger.error(f"update_visuals error: {e}", exc_info=True)
        # Show error figs
        fig_err = go.Figure()
        is_dark = (theme_value == [1])
        if is_dark:
            fig_err.update_layout(**get_plotly_dark_layout(f"Error: {e}"))
        else:
            fig_err.update_layout(**get_plotly_light_layout(f"Error: {e}"))
        return fig_err, fig_err, fig_err, fig_err, fig_err, []

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
