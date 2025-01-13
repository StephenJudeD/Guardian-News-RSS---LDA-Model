#!/usr/bin/env python
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
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Stop words (Expanded)
# ─────────────────────────────────────────────────────────────────────
CUSTOM_STOP_WORDS = {
    'says', 'said', 'would', 'also', 'one', 'new', 'like', 'get', 'make',
    'first', 'two', 'year', 'years', 'time', 'way', 'says', 'say', 'saying',
    'according', 'told', 'reuters', 'guardian', 'monday', 'tuesday', 'wednesday',
    'thursday', 'friday', 'saturday', 'sunday', 'week', 'month', 'us', 'people',
    'government', 'could', 'will', 'may', 'trump', 'published', 'article',
    'editor', 'nt', 'dont', 'doesnt', 'cant', 'couldnt', 'shouldnt'
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
    Fetch articles from The Guardian, filter by date,
    tokenize, detect bigrams/trigrams, then train LDA on the ENTIRE set.
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

        # Build bigrams & trigrams
        bigram_phrases = Phrases(tokenized_texts, min_count=5, threshold=10)
        trigram_phrases = Phrases(bigram_phrases[tokenized_texts], threshold=10)
        bigram = Phraser(bigram_phrases)
        trigram = Phraser(trigram_phrases)

        texts = []
        for t in tokenized_texts:
            bigrammed = bigram[t]
            trigrammed = trigram[bigrammed]
            texts.append(trigrammed)

        # Gensim dictionary + corpus
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
    Bubble chart: doc length vs published date, sized by doc length, colored by dominant_topic.
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
            size='doc_length',
            color='dominant_topic',
            hover_data=['title'],
            title='Document Length Bubble Chart'
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=f"Bubble Chart Error: {e}")


def create_ngram_bar_chart(texts):
    """
    Bar chart of the most common bigrams/trigrams (indicated by underscores).
    We'll pick the top 15 by frequency.
    """
    try:
        ngram_counts = {}
        for tokens in texts:
            for tok in tokens:
                if "_" in tok:
                    ngram_counts[tok] = ngram_counts.get(tok, 0) + 1

        if not ngram_counts:
            return go.Figure().update_layout(
                template='plotly',
                title="No bigrams/trigrams found"
            )

        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        top_ngrams = sorted_ngrams[:15]
        df_ngram = pd.DataFrame(top_ngrams, columns=["ngram", "count"])

        fig = px.bar(
            df_ngram,
            x="count",
            y="ngram",
            orientation="h",
            title="Top Bigrams & Trigrams"
        )
        fig.update_layout(template='plotly', yaxis={"categoryorder": "total ascending"})
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram bar chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=f"Ngram Bar Chart Error: {e}")


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
                        style={"color": "white", "fontWeight": "bold", "fontSize": "2.4rem"}
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
        # Date Range Card
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
            md=3
        ),
        # Topic Filter
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
            md=3
        ),
        # Section Filter (empty for now, updated in callback)
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Select Sections", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='section-filter',
                            # Will be dynamically updated by callback
                            options=[],  # placeholder
                            multi=True,
                            placeholder="Filter by sections...",
                            className="mb-2"
                        )
                    ]),
                ],
                className="mb-2",
                style={"backgroundColor": "white"}
            ),
            md=3
        ),
        dbc.Col(md=3)  # blank space
    ],
    className="my-2 px-2"
)

topic_dist_card = dbc.Card(
    [
        dbc.CardHeader("Topic Word Distributions", style={"backgroundColor": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Loading(
                id="loading-topic-dist",
                type="circle",
                children=[dcc.Graph(id='topic-distribution', style={"height": "600px"})]
            ),
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
            dcc.Loading(
                id="loading-3d-tsne",
                type="circle",
                children=[dcc.Graph(id='tsne-plot', style={"height": "600px"})]
            ),
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
            dcc.Loading(
                id="loading-wordcloud",
                type="circle",
                children=[dcc.Graph(id='word-cloud', style={"height": "600px"})]
            ),
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
            dcc.Loading(
                id="loading-bubble-chart",
                type="circle",
                children=[dcc.Graph(id='bubble-chart', style={"height": "600px"})]
            ),
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

bigrams_trigrams_card = dbc.Card(
    [
        dbc.CardHeader("Bigrams & Trigrams", style={"backgroundColor": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Loading(
                id="loading-bigrams-trigrams",
                type="circle",
                children=[dcc.Graph(id='bigrams-trigrams', style={"height": "600px"})]
            ),
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
    dbc.Row([dbc.Col(bigrams_trigrams_card, md=12)], className="g-3"),
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
        # default
        start_date = end_date - timedelta(days=30)
    return start_date, end_date


@app.callback(
    Output('section-filter', 'options'),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ]
)
def update_section_filter_options(start_date, end_date):
    """
    Dynamically fetch unique sections from the DF to populate section dropdown.
    """
    df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date)
    if df is None or df.empty:
        return []
    unique_sections = sorted(df['section'].dropna().unique())
    return [{"label": s, "value": s} for s in unique_sections]


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
        Input('topic-filter', 'value'),
        Input('section-filter', 'value')
    ]
)
def update_visuals(start_date, end_date, selected_topics, selected_sections):
    """
    Main callback: 
    1) Train LDA on entire set (already done in process_articles).
    2) Filter DF by chosen sections (if any).
    3) Build & filter for topic visuals (if topics are chosen).
    4) Return updated figures & table data.
    """
    try:
        logger.info(f"update_visuals: {start_date} to {end_date}, topics={selected_topics}, sections={selected_sections}")

        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date)
        if df is None or df.empty:
            empty_fig = go.Figure().update_layout(template='plotly', title="No Data")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

        # If user selected some sections, we filter df + texts + corpus
        # NOTE: This does NOT retrain LDA on the subset. Topics remain from entire corpus.
        if selected_sections:
            # Filter df
            mask = df['section'].isin(selected_sections)
            df_filtered = df[mask].copy()
            # We also need to filter texts and corpus accordingly
            texts_filtered = []
            corpus_filtered = []
            idx_map = []
            for i, row in enumerate(df.index):
                if row in df_filtered.index:
                    texts_filtered.append(texts[i])
                    corpus_filtered.append(corpus[i])
                    idx_map.append(row)
            # reindex df_filtered
            df_filtered.reset_index(drop=True, inplace=True)
        else:
            df_filtered = df.copy()
            texts_filtered = texts
            corpus_filtered = corpus
            idx_map = list(df.index)

        if df_filtered.empty:
            empty_fig = go.Figure().update_layout(template='plotly', title="No Data (No sections match)")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

        # If no topic selected, default to [0..4]
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
        # We must pass the *filtered* df and corpus to the TSNE function
        # but that function references doc index in the original df => we must adapt it
        # For simplicity, let's clone the code so it can handle a filtered df:
        if len(df_filtered) < 2:
            tsne_fig = go.Figure().update_layout(
                template='plotly',
                title='Not enough docs for t-SNE (sections filter)'
            )
        else:
            doc_topics_list = []
            # reconstruct doc_topics_list from corpus_filtered
            for c in corpus_filtered:
                # c is doc2bow, so we do lda_model[c]
                topic_weights = [0.0]*lda_model.num_topics
                for topic_id, w in lda_model[c]:
                    topic_weights[topic_id] = w
                doc_topics_list.append(topic_weights)

            doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
            if len(doc_topics_array) < 2:
                tsne_fig = go.Figure().update_layout(
                    template='plotly',
                    title='Not enough docs for t-SNE (sections filter)'
                )
            else:
                perplex_val = 30
                if len(doc_topics_array) < 30:
                    perplex_val = max(2, len(doc_topics_array) - 1)

                tsne = TSNE(n_components=3, random_state=42, perplexity=perplex_val, n_jobs=1)
                embedded = tsne.fit_transform(doc_topics_array)

                # We'll build a small df for the scatter
                scatter_rows = []
                for row_i, doc_arr in enumerate(doc_topics_array):
                    # row_i is in [0..len(df_filtered)-1], but we need the actual doc index from idx_map
                    original_idx = idx_map[row_i]
                    # find the row in df_filtered that corresponds
                    # but we have it by direct indexing
                    scatter_rows.append({
                        'x': embedded[row_i, 0],
                        'y': embedded[row_i, 1],
                        'z': embedded[row_i, 2],
                        'dominant_topic': int(np.argmax(doc_arr)),
                        'title': df.at[original_idx, 'title']
                    })

                scatter_df = pd.DataFrame(scatter_rows)
                tsne_fig = px.scatter_3d(
                    scatter_df,
                    x='x', y='y', z='z',
                    color='dominant_topic',
                    hover_data=['title'],
                    title='3D t-SNE Topic Clustering'
                )
                tsne_fig.update_layout(template='plotly')

        # 4) Bubble Chart
        doc_lengths = []
        doc_dominant_topics = []
        for i, row_idx in enumerate(df_filtered.index):
            c = corpus_filtered[i]
            text_toks = texts_filtered[i]
            doc_topics = lda_model.get_document_topics(c)
            n_tokens = len(text_toks) if text_toks else 0
            doc_lengths.append(n_tokens)
            if doc_topics:
                best_t = max(doc_topics, key=lambda x: x[1])[0]
            else:
                best_t = -1
            doc_dominant_topics.append(best_t)

        df_filtered["doc_length"] = doc_lengths
        df_filtered["dominant_topic"] = doc_dominant_topics
        bubble_fig = create_bubble_chart(df_filtered)

        # 5) Bigrams & Trigrams bar chart
        ngram_fig = create_ngram_bar_chart(texts_filtered)

        # 6) Article Table
        table_data = []
        for i, row_idx in enumerate(df_filtered.index):
            doc_topics = lda_model.get_document_topics(corpus_filtered[i])
            these_topics = []
            for (tid, w) in sorted(doc_topics, key=lambda x: x[1], reverse=True):
                if tid in selected_topics:
                    these_topics.append(f"Topic {tid}: {w:.3f}")
            table_data.append({
                'title': df_filtered.at[i, 'title'],
                'section': df_filtered.at[i, 'section'],
                'published': df_filtered.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(these_topics)
            })

        return dist_fig, wc_fig, tsne_fig, bubble_fig, ngram_fig, table_data

    except Exception as e:
        logger.error(f"update_visuals error: {e}", exc_info=True)
        empty_fig = go.Figure().update_layout(template='plotly', title=f"Error: {e}")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
