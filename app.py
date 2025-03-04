import dash
from dash import Dash, html, dcc, Input, Output, dash_table
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
    'nt', 'dont', 'doesnt', 'cant', 'couldnt', 'shouldnt'
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
# Dash Setup - IMPROVED THEME
# ─────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css'])
server = app.server
app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────
# ENHANCED COLORS - Guardian Brand Colors
# ─────────────────────────────────────────────────────────────────────
COLORS = {
    'navy': '#052962',
    'blue': '#0084C6',
    'red': '#C70000',
    'yellow': '#FFBB50',
    'light_blue': '#00B2FF',
    'gray': '#F6F6F6',
    'dark_gray': '#333333',
    'white': '#FFFFFF'
}

# ─────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def process_articles(start_date, end_date):
    """
    Fetch Guardian articles in the given date range,
    then tokenize, detect bigrams/trigrams, and train LDA on the entire set.
    Return (df, texts, dictionary, corpus, lda_model).
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
# NEW FUNCTION: Generate Topic Labels
# ─────────────────────────────────────────────────────────────────────
def generate_topic_labels(lda_model, n_terms=3):
    """Generate human-readable labels for topics using top terms"""
    labels = []
    for topic_id in range(lda_model.num_topics):
        terms = [term for term, _ in lda_model.show_topic(topic_id, topn=n_terms)]
        label = " | ".join(terms[:3])
        labels.append(f"Topic {topic_id}: {label.title()}")
    return labels

# ─────────────────────────────────────────────────────────────────────
# Visualization Helpers - IMPROVED VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────
def create_word_cloud(topic_words):
    """
    Word cloud from LDA topic-word pairs, improved aesthetics.
    """
    try:
        freq_dict = dict(topic_words)
        wc = WordCloud(
            background_color=COLORS['white'],
            width=1000,
            height=500,
            colormap='Blues',
            max_words=100,
            contour_width=1,
            contour_color=COLORS['navy']
        ).generate_from_frequencies(freq_dict)
        fig = px.imshow(wc)
        fig.update_layout(
            template='plotly_white', 
            title="Topic Word Cloud",
            margin=dict(l=20, r=20, t=40, b=20),
            title_font=dict(size=22, color=COLORS['navy']),
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly_white')

def create_tsne_visualization_3d(df, corpus, lda_model):
    """
    3D t-SNE scatter (Plotly) with improved aesthetics.
    Uses all documents in df/corpus to avoid filtering by topic.
    """
    try:
        if df is None or len(df) < 2:
            return go.Figure().update_layout(
                template='plotly_white',
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
                template='plotly_white',
                title='Not enough docs for t-SNE'
            )

        perplex_val = 30
        if len(doc_topics_array) < 30:
            perplex_val = max(2, len(doc_topics_array) - 1)

        tsne = TSNE(
            n_components=3,
            random_state=42,
            perplexity=perplex_val,
            n_jobs=1
        )
        embedded = tsne.fit_transform(doc_topics_array)  # shape: (N, 3)

        # Get topic labels for better display
        topic_labels = generate_topic_labels(lda_model)
        topic_label_map = {i: label for i, label in enumerate(topic_labels)}

        scatter_df = pd.DataFrame({
            'x': embedded[:, 0],
            'y': embedded[:, 1],
            'z': embedded[:, 2],
            'dominant_topic': [np.argmax(row) for row in doc_topics_array],
            'topic_label': [topic_label_map[np.argmax(row)] for row in doc_topics_array],
            'doc_index': df.index,
            'title': df['title']
        })

        fig = px.scatter_3d(
            scatter_df,
            x='x', y='y', z='z',
            color='dominant_topic',
            hover_data=['title', 'topic_label'],
            title='3D Topic Clustering',
            color_continuous_scale=px.colors.sequential.Blues,
            labels={'dominant_topic': 'Topic'}
        )
        
        fig.update_layout(
            template='plotly_white',
            title_font=dict(size=22, color=COLORS['navy']),
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                bgcolor=COLORS['gray']
            ),
            margin=dict(l=0, r=0, t=45, b=0)
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating 3D t-SNE: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly_white', title=str(e))

def create_bubble_chart(df):
    """
    Bubble chart: doc length vs published date, sized by doc length,
    colored by dominant_topic with improved aesthetics.
    """
    try:
        if df is None or df.empty:
            return go.Figure().update_layout(
                template='plotly_white',
                title='Bubble Chart Unavailable'
            )

        # Get topic labels
        topic_counts = df['dominant_topic'].value_counts().to_dict()
        topic_labels = [f"Topic {i}" for i in range(max(topic_counts.keys())+1)]
        
        fig = px.scatter(
            df,
            x='published',
            y='doc_length',
            size='doc_length',
            size_max=35,
            color='dominant_topic',
            color_continuous_scale=px.colors.sequential.Blues,
            hover_data=['title'],
            title='Article Length Over Time',
            labels={
                'published': 'Publication Date',
                'doc_length': 'Word Count',
                'dominant_topic': 'Topic'
            }
        )
        
        fig.update_layout(
            template='plotly_white',
            title_font=dict(size=22, color=COLORS['navy']),
            plot_bgcolor=COLORS['gray'],
            xaxis=dict(
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='white'
            ),
            yaxis=dict(
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='white'
            ),
            coloraxis_colorbar=dict(
                title='Topic',
                tickvals=list(range(len(topic_labels))),
                ticktext=topic_labels
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly_white', title=str(e))

def create_ngram_bar_chart(texts):
    """
    Bar chart of the most common bigrams/trigrams with improved aesthetics.
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
                template='plotly_white',
                title="No bigrams/trigrams found"
            )

        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        top_ngrams = sorted_ngrams[:15]
        df_ngram = pd.DataFrame(top_ngrams, columns=["ngram", "count"])
        
        # Format ngrams to look better (replace underscores with spaces)
        df_ngram["display_ngram"] = df_ngram["ngram"].apply(lambda x: x.replace("_", " "))

        fig = px.bar(
            df_ngram,
            x="count",
            y="display_ngram",
            orientation="h",
            title="Top Multi-Word Phrases",
            color="count",
            color_continuous_scale=px.colors.sequential.Blues,
            labels={"count": "Frequency", "display_ngram": "Phrase"}
        )
        
        fig.update_layout(
            template='plotly_white',
            title_font=dict(size=22, color=COLORS['navy']),
            plot_bgcolor=COLORS['gray'],
            xaxis=dict(
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='white'
            ),
            yaxis=dict(
                title_font=dict(size=14),
                tickfont=dict(size=12),
                categoryorder="total ascending"
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram bar chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly_white', title=str(e))

# ─────────────────────────────────────────────────────────────────────
# NEW: Topic-Term Heatmap Visualization
# ─────────────────────────────────────────────────────────────────────
def create_term_topic_heatmap(lda_model, topic_labels, topn=10):
    """
    Create a heatmap showing term distribution across topics
    """
    try:
        # Get top terms for each topic
        topic_terms = []
        for topic_id in range(lda_model.num_topics):
            terms = lda_model.show_topic(topic_id, topn=topn)
            for term, prob in terms:
                topic_terms.append({
                    "Topic": topic_labels[topic_id],
                    "Term": term,
                    "Probability": prob
                })
        
        df = pd.DataFrame(topic_terms)
        
        # Create heatmap
        fig = px.density_heatmap(
            df,
            x="Topic",
            y="Term",
            z="Probability",
            title="Term Distribution Across Topics",
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(
            template='plotly_white',
            title_font=dict(size=22, color=COLORS['navy']),
            xaxis=dict(title="", tickangle=45),
            yaxis=dict(title="", categoryorder="total ascending"),
            margin=dict(l=20, r=20, t=40, b=100)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating term-topic heatmap: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly_white', title=str(e))

# ─────────────────────────────────────────────────────────────────────
# IMPROVED UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────
# Card style for consistent look
card_style = {
    "border": "none", 
    "borderRadius": "8px", 
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "transition": "transform 0.3s",
    "backgroundColor": COLORS['white']
}

card_header_style = {
    "backgroundColor": COLORS['navy'], 
    "color": COLORS['white'],
    "fontWeight": "bold",
    "borderTopLeftRadius": "8px",
    "borderTopRightRadius": "8px",
    "padding": "12px 16px"
}

# Navbar with improved styling
navbar = dbc.Navbar(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.I(className="fas fa-newspaper", style={"fontSize": "2.2rem", "color": "white"}), 
                    width="auto"
                ),
                dbc.Col(
                    dbc.NavbarBrand(
                        "Guardian News Topic Explorer",
                        className="ms-2",
                        style={"color": "white", "fontWeight": "bold", "fontSize": "1.8rem"}
                    )
                )
            ],
            align="center",
            className="g-0",
        )
    ],
    color=COLORS['navy'],
    dark=True,
    className="mb-3 px-4 py-2",
    style={"boxShadow": "0 2px 4px rgba(0,0,0,0.15)"}
)

# ─────────────────────────────────────────────────────────────────────
# About This App (Explainer) - ENHANCED
# ─────────────────────────────────────────────────────────────────────
explainer_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-info-circle me-2"),
                "About This App"
            ],
            style=card_header_style
        ),
        dbc.CardBody(
            [
                html.P(
                    [
                        "This dashboard fetches articles from the Guardian's API, processes them with "
                        "Natural Language Processing (NLP), and then applies techniques like LDA for topic modeling, "
                        "bigrams/trigrams detection for multi-word phrases, and t-SNE for visualizing clusters in 3D. "
                        "Explore the date range and topic filters to see how news stories shift over time! ",
                        html.A(
                            [
                                html.I(className="fab fa-github me-1"),
                                "Code & Readme available on GitHub"
                            ],
                            href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model/tree/main",
                            target="_blank",
                            style={"color": COLORS['blue'], "textDecoration": "underline", "fontWeight": "bold"},
                        ),
                    ],
                    className="mb-0",
                    style={"fontSize": "1.1rem", "lineHeight": "1.6"}
                )
            ],
            style={"backgroundColor": COLORS['white']},
        ),
    ],
    className="mb-4",
    style=card_style
)

# Filters with enhanced UI
controls_row = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.I(className="far fa-calendar-alt me-2"),
                            "Select Date Range"
                        ],
                        style=card_header_style
                    ),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id='date-select-buttons',
                            options=[
                                {'label': 'Last Day', 'value': 'last_day'},
                                {'label': 'Last Week', 'value': 'last_week'},
                                {'label': 'Last Two Weeks', 'value': 'last_two_weeks'},
                            ],
                            value='last_two_weeks',
                            inline=True,
                            className="mb-3",
                            inputStyle={"margin-right": "5px"},
                            labelStyle={"margin-right": "15px", "font-weight": "500"}
                        ),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=(datetime.now() - timedelta(days=14)).date(),
                            end_date=datetime.now().date(),
                            className="mb-2",
                            style={"width": "100%"}
                        )
                    ]),
                ],
                className="mb-3",
                style=card_style
            ),
            md=5
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.I(className="fas fa-filter me-2"),
                            "Select Topics"
                        ],
                        style=card_header_style
                    ),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='topic-filter',
                            options=[{'label': f'Topic {i}', 'value': i} for i in range(5)],
                            multi=True,
                            placeholder="Filter by topics...",
                            className="mb-2",
                            style={"width": "100%"}
                        ),
                        html.Div(
                            [
                                html.I(className="fas fa-lightbulb text-warning me-2"),
                                "Topics will update with descriptive labels after processing"
                            ],
                            className="text-muted small mt-2"
                        )
                    ]),
                ],
                className="mb-3",
                style=card_style
            ),
            md=5
        ),
        dbc.Col(md=2)  # blank space
    ],
    className="my-3 px-3"
)

# IMPROVED CARDS
topic_dist_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-chart-bar me-2"),
                "Topic Word Distributions"
            ],
            style=card_header_style
        ),
        dbc.CardBody(
            dbc.Spinner(
                dcc.Graph(id='topic-distribution', style={"height": "600px"}),
                color=COLORS['blue'],
                size="lg",
                type="border"
            ),
            style={"backgroundColor": COLORS['white']}
        )
    ],
    className="mb-4",
    style=card_style
)

tsne_3d_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-cube me-2"),
                "3D Topic Clustering",
                html.I(
                    className="fas fa-question-circle ms-2",
                    id="tsne-info",
                    style={"color": COLORS['light_blue'], "cursor": "pointer"}
                )
            ],
            style=card_header_style
        ),
        dbc.Tooltip(
            "Drag to rotate, scroll to zoom. Each point is an article positioned by topic similarity.",
            target="tsne-info",
            placement="top"
        ),
        dbc.CardBody(
            dbc.Spinner(
                dcc.Graph(id='tsne-plot', style={"height": "600px"}),
                color=COLORS['blue'],
                size="lg",
                type="border"
            ),
            style={"backgroundColor": COLORS['white']}
        )
    ],
    className="mb-4",
    style=card_style
)

wordcloud_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-cloud me-2"),
                "Word Cloud"
            ],
            style=card_header_style
        ),
        dbc.CardBody(
            dbc.Spinner(
                dcc.Graph(id='word-cloud', style={"height": "600px"}),
                color=COLORS['blue'],
                size="lg",
                type="border"
            ),
            style={"backgroundColor": COLORS['white']}
        )
    ],
    className="mb-4",
    style=card_style
)

bubble_chart_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-chart-scatter me-2"),
                "Article Length Over Time"
            ],
            style=card_header_style
        ),
        dbc.CardBody(
            dbc.Spinner(
                dcc.Graph(id='bubble-chart', style={"height": "600px"}),
                color=COLORS['blue'],
                size="lg",
                type="border"
            ),
            style={"backgroundColor": COLORS['white']}
        )
    ],
    className="mb-4",
    style=card_style
)

bigrams_trigrams_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-chart-line me-2"),
                "Top Multi-Word Phrases"
            ],
            style=card_header_style
        ),
        dbc.CardBody(
            dbc.Spinner(
                dcc.Graph(id='bigrams-trigrams', style={"height": "600px"}),
                color=COLORS['blue'],
                size="lg",
                type="border"
            ),
            style={"backgroundColor": COLORS['white']}
        )
    ],
    className="mb-4",
    style=card_style
)

# NEW CARD: Term-Topic Heatmap
heatmap_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-th me-2"),
                "Term-Topic Distribution"
            ],
            style=card_header_style
        ),
        dbc.CardBody(
            dbc.Spinner(
                dcc.Graph(id='term-topic-heatmap', style={"height": "600px"}),
                color=COLORS['blue'],
                size="lg",
                type="border"
            ),
            style={"backgroundColor": COLORS['white']}
        )
    ],
    className="mb-4",
    style=card_style
)

article_table_card = dbc.Card(
    [
        dbc.CardHeader(
            [
                html.I(className="fas fa-newspaper me-2"),
                "Article Details"
            ],
            style=card_header_style
        ),
        dbc.CardBody([
            dash_table.DataTable(
                id='article-details',
                columns=[
                    {'name': 'Title', 'id': 'title'},
                    {'name': 'Published', 'id': 'published'},
                    {'name': 'Topics', 'id': 'topics'},
                ],
                style_table={
                    'overflowX': 'auto',
                    'border': '1px solid #eee',
                    'borderRadius': '5px'
                },
                style_cell={
                    'backgroundColor': 'white',
                    'color': 'black',
                    'textAlign': 'left',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'padding': '10px',
                    'fontSize': '14px',
                    'fontFamily': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': COLORS['navy'],
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'left',
                    'padding': '12px 10px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': COLORS['gray']
                    },
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': '#e1f5fe',
                        'border': '1px solid #2196f3'
                    }
                ],
                page_size=10,
                page_action='native',
                sort_action='native',
                sort_mode='multi'
            )
        ], style={"backgroundColor": COLORS['white']})
    ],
    className="mb-4",
    style=card_style
)

# IMPROVED LAYOUT
app.layout = dbc.Container([
    navbar,
    dbc.Row([dbc.Col(explainer_card, md=12)], className="g-3"),
    controls_row,
    dbc.Row([dbc.Col(topic_dist_card, md=12)], className="g-3"),
    dbc.Row(
        [
            dbc.Col(tsne_3d_card, md=6),
            dbc.Col(wordcloud_card, md=6)
        ], 
        className="g-3"
    ),
    dbc.Row(
        [
            dbc.Col(bubble_chart_card, md=6),
            dbc.Col(bigrams_trigrams_card, md=6)
        ], 
        className="g-3"
    ),
    dbc.Row([dbc.Col(heatmap_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(article_table_card, md=12)], className="g-3"),
], fluid=True, style={"backgroundColor": "#f5f7fa", "padding": "20px"})

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
    elif selected_range == 'last_two_weeks':
        start_date = end_date - timedelta(days=14)
    else:
        start_date = end_date - timedelta(days=14)
    return start_date, end_date


@app.callback(
    [
        Output('topic-distribution', 'figure'),
        Output('word-cloud', 'figure'),
        Output('tsne-plot', 'figure'),
        Output('bubble-chart', 'figure'),
        Output('bigrams-trigrams', 'figure'),
        Output('term-topic-heatmap', 'figure'),
        Output('article-details', 'data'),
        Output('topic-filter', 'options')
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
     1) Train LDA on entire set within the selected date range.
     2) If no topic selected, show all topics [0..4].
     3) Build visuals & article table.
     4) We now create the t-SNE from the entire set (unfiltered).
    """
    try:
        logger.info(f"update_visuals: {start_date} to {end_date}, topics={selected_topics}")
        
        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date)
        if df is None or df.empty:
            empty_fig = go.Figure().update_layout(
                template='plotly_white', 
                title="No Data Available",
                title_font=dict(size=22, color=COLORS['navy'])
            )
            empty_options = [{'label': f'Topic {i}', 'value': i} for i in range(5)]
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], empty_options

        # Generate human-readable topic labels
        topic_labels = generate_topic_labels(lda_model)
        dropdown_options = [{'label': lbl, 'value': i} for i, lbl in enumerate(topic_labels)]

        # If no topic selected, default to [0 .. 4]
        if not selected_topics:
            selected_topics = list(range(lda_model.num_topics))

        # Build doc-level info: doc_length, dominant_topic
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

        # Filter out articles whose dominant_topic is not in selected_topics
        filtered_df = df[df["dominant_topic"].isin(selected_topics)].copy()
        if filtered_df.empty:
            # means user selected topics not present in the data
            empty_fig = go.Figure().update_layout(
                template='plotly_white', 
                title="No Articles Found For Selected Topics",
                title_font=dict(size=22, color=COLORS['navy'])
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], dropdown_options

        # Build subset of texts/corpus for the articles in filtered_df
        filtered_texts = []
        filtered_corpus = []
        for i in filtered_df.index:
            filtered_texts.append(texts[i])
            filtered_corpus.append(corpus[i])

        # 1) Topic Word Distribution (for all selected topics)
        words_list = []
        for t_id in selected_topics:
            top_pairs = lda_model.show_topic(t_id, topn=20)
            for (w, prob) in top_pairs:
                words_list.append((w, prob, topic_labels[t_id]))
        if not words_list:
            dist_fig = go.Figure().update_layout(
                template='plotly_white', 
                title="No topics found",
                title_font=dict(size=22, color=COLORS['navy'])
            )
        else:
            df_dist = pd.DataFrame(words_list, columns=["word", "prob", "topic"])
            dist_fig = px.bar(
                df_dist,
                x="prob",
                y="word",
                color="topic",
                orientation="h",
                title="Topic Word Distributions",
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"prob": "Probability", "word": "Term", "topic": "Topic"}
            )
            dist_fig.update_layout(
                template='plotly_white',
                title_font=dict(size=22, color=COLORS['navy']),
                plot_bgcolor=COLORS['gray'],
                yaxis={'categoryorder': 'total ascending'},
                xaxis=dict(
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    gridcolor='white'
                ),
                yaxis=dict(
                    title_font=dict(size=14),
                    tickfont=dict(size=12)
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

        # 2) Word Cloud (take first selected topic)
        first_topic = selected_topics[0]
        wc_fig = create_word_cloud(lda_model.show_topic(first_topic, topn=30))

        # 3) 3D t-SNE (use the entire df/corpus, not the filtered_df/corpus)
        tsne_fig = create_tsne_visualization_3d(df, corpus, lda_model)

        # 4) Bubble Chart (use filtered_df)
        bubble_fig = create_bubble_chart(filtered_df)

        # 5) Bigrams & Trigrams (use filtered texts)
        ngram_fig = create_ngram_bar_chart(filtered_texts)
        
        # 6) Term-Topic Heatmap
        heatmap_fig = create_term_topic_heatmap(lda_model, topic_labels)

        # 7) Article Table data (only for the filtered set)
        table_data = []
        for i in filtered_df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            these_topics = [
                f"{topic_labels[tid]} ({w:.3f})" 
                for (tid, w) in sorted(doc_topics, key=lambda x: x[1], reverse=True)
                if tid in selected_topics
            ]
            table_data.append({
                'title': filtered_df.at[i, 'title'],
                'published': filtered_df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(these_topics[:2])  # Show top 2 topics
            })

        return dist_fig, wc_fig, tsne_fig, bubble_fig, ngram_fig, heatmap_fig, table_data, dropdown_options

    except Exception as e:
        logger.error(f"update_visuals error: {e}", exc_info=True)
        empty_fig = go.Figure().update_layout(
            template='plotly_white', 
            title=f"Error: {e}",
            title_font=dict(size=22, color=COLORS['red'])
        )
        empty_options = [{'label': f'Topic {i}', 'value': i} for i in range(5)]
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], empty_options


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
