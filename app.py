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
    # Add as many extra items as you want:
    'breaking', 'update', 'live', 'say'
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
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────
# Guardian Theme Plot Layout Helper
# ─────────────────────────────────────────────────────────────────────
def get_guardian_plot_layout(fig_title=""):
    """Return a default layout for Guardian-themed figures."""
    return dict(
        paper_bgcolor="white",
        plot_bgcolor="#f6f6f6",
        font=dict(family="Guardian Egyptian Web, Georgia, serif"),
        title=fig_title,
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(family="Guardian Egyptian Web, Georgia, serif", size=18, color="#005689"),
        legend_title_font=dict(family="Guardian Egyptian Web, Georgia, serif", size=12),
        legend_font=dict(family="Guardian Egyptian Web, Georgia, serif", size=10),
        colorway=["#005689", "#c70000", "#ffbb00", "#00b2ff", "#90dcff", "#ff5b5b", 
                  "#4bc6df", "#aad801", "#43853d", "#767676"],
        xaxis=dict(
            gridcolor="#dcdcdc",
            zerolinecolor="#dcdcdc",
            showgrid=True,
            showline=True,
            linecolor="#dcdcdc"
        ),
        yaxis=dict(
            gridcolor="#dcdcdc",
            zerolinecolor="#dcdcdc",
            showgrid=True,
            showline=True,
            linecolor="#dcdcdc"
        ),
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
def create_word_cloud(topic_words):
    """
    Create a word cloud from LDA topic-word pairs using Guardian colors.
    """
    try:
        freq_dict = dict(topic_words)
        
        wc = WordCloud(
            background_color="white",
            width=800,
            height=400,
            colormap="Blues",  # Use blues to match Guardian color scheme
            max_words=50,
            prefer_horizontal=0.9
        ).generate_from_frequencies(freq_dict)

        fig = px.imshow(wc)
        fig.update_layout(**get_guardian_plot_layout("Topic Word Cloud"))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating word cloud: {e}"))
        return fig

def create_tsne_visualization_3d(df, corpus, lda_model, perplexity=30):
    """
    3D t-SNE scatter (Plotly).
    Uses all documents in df/corpus to avoid filtering by topic.
    Allows dynamic perplexity setting.
    """
    try:
        if df is None or len(df) < 2:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("Not enough documents for t-SNE"))
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
            fig.update_layout(**get_guardian_plot_layout("Not enough docs for t-SNE"))
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
        fig.update_layout(**get_guardian_plot_layout())
        return fig
    except Exception as e:
        logger.error(f"Error creating 3D t-SNE: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating 3D t-SNE: {e}"))
        return fig

def create_bubble_chart(df):
    """
    Bubble chart: doc length vs published date, sized by doc length,
    colored by dominant_topic. Removes outliers & uses log scale.
    """
    try:
        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("Bubble Chart Unavailable"))
            return fig

        cut_off = df['doc_length'].quantile(0.95)
        filtered_df = df[df['doc_length'] <= cut_off].copy()
        if filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("No Data after outlier removal"))
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
        fig.update_layout(**get_guardian_plot_layout())
        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating bubble chart: {e}"))
        return fig

def create_ngram_radar_chart(texts):
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
            fig.update_layout(**get_guardian_plot_layout("No bigrams/trigrams found"))
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
        fig.update_layout(**get_guardian_plot_layout())
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram radar chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating ngram radar chart: {e}"))
        return fig

# ─────────────────────────────────────────────────────────────────────
# Guardian Theme CSS
# ─────────────────────────────────────────────────────────────────────
guardian_theme_css = html.Div([
    html.Style('''
        /* Guardian Theme CSS */
        :root {
            --guardian-blue: #005689;
            --guardian-blue-light: #00b2ff;
            --guardian-red: #c70000;
            --guardian-yellow: #ffbb00;
            --guardian-bg: #f6f6f6;
            --guardian-border: #dcdcdc;
        }
        
        body {
            font-family: "Guardian Egyptian Web", Georgia, serif;
            background-color: var(--guardian-bg);
        }
        
        .guardian-navbar {
            background-color: var(--guardian-blue);
            color: white;
            padding: 10px 0;
            border-bottom: 2px solid var(--guardian-yellow);
        }
        
        .guardian-card {
            border: 1px solid var(--guardian-border);
            border-radius: 2px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            background-color: white;
            margin-bottom: 20px;
        }
        
        .guardian-header {
            background-color: var(--guardian-blue);
            color: white;
            font-weight: bold;
            padding: 12px 15px;
            border-bottom: 2px solid var(--guardian-yellow);
        }
        
        .guardian-card-body {
            padding: 20px;
            background-color: white;
        }
        
        /* Control styles */
        .guardian-control {
            background-color: white;
            border: 1px solid var(--guardian-border);
            padding: 15px;
            border-radius: 2px;
        }
        
        /* Grid for layout */
        .guardian-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
    ''')
])

# ─────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src="https://static.guim.co.uk/sys-images/Guardian/Pix/pictures/2010/03/01/poweredbyguardianBLACK.png", 
                                height="32px", className="mr-2"),
                        width="auto",
                    ),
                    dbc.Col(
                        html.H3("Guardian News Topic Explorer", className="mb-0 text-white"),
                        width="auto",
                    ),
                ],
                align="center",
                className="g-0",
            ),
        ],
        fluid=True
    ),
    className="guardian-navbar mb-4",
    dark=True
)

explainer_card = dbc.Card(
    [
        dbc.CardHeader("About This App", className="guardian-header"),
        dbc.CardBody(
            [
                html.P(
                    [
                        "This dashboard fetches articles from the Guardian's RSS, processes them with "
                        "Natural Language Processing (NLP), and then applies techniques like LDA for topic modeling, "
                        "bigrams/trigrams detection for multi-word phrases, and t-SNE for visualizing clusters in 3D. "
                        "Use the controls below to explore the data: date range, dynamic LDA topic counts, perplexity, "
                        "and more. See how news stories shift over time! ",
                        html.A(
                            "Code & Readme available @ GitHub",
                            href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model/tree/main",
                            target="_blank",
                            style={"textDecoration": "underline", "color": "#005689"},
                        ),
                    ],
                    className="mb-0",
                )
            ],
            className="guardian-card-body"
        ),
    ],
    className="mb-3 guardian-card",
)

date_filter_card = dbc.Card(
    [
        dbc.CardHeader("Select Date Range", className="guardian-header"),
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
            ),
            dcc.DatePickerRange(
                id='date-range',
                start_date=(datetime.now() - timedelta(days=7)).date(),
                end_date=datetime.now().date()
            )
        ], className="guardian-card-body"),
    ],
    className="mb-2 guardian-card",
)

num_topics_card = dbc.Card(
    [
        dbc.CardHeader("LDA: Number of Topics", className="guardian-header"),
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
        ], className="guardian-card-body"),
    ],
    className="mb-2 guardian-card",
)

tsne_controls_card = dbc.Card(
    [
        dbc.CardHeader("t-SNE Perplexity", className="guardian-header"),
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
        ], className="guardian-card-body"),
    ],
    className="mb-2 guardian-card",
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
        dbc.CardHeader("Topic Word Distributions (Top 10)", className="guardian-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-topic-dist",
                type="circle",
                children=[dcc.Graph(id='topic-distribution', style={"height": "600px"})]
            ),
            className="guardian-card-body"
        )
    ],
    className="mb-3 guardian-card",
)

tsne_3d_card = dbc.Card(
    [
        dbc.CardHeader("3D t-SNE Topic Clustering", className="guardian-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-3d-tsne",
                type="circle",
                children=[dcc.Graph(id='tsne-plot', style={"height": "600px"})]
            ),
            className="guardian-card-body"
        )
    ],
    className="mb-3 guardian-card",
)

wordcloud_card = dbc.Card(
    [
        dbc.CardHeader("Word Cloud", className="guardian-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-wordcloud",
                type="circle",
                children=[dcc.Graph(id='word-cloud', style={"height": "600px"})]
            ),
            className="guardian-card-body"
        )
    ],
    className="mb-3 guardian-card",
)

bubble_chart_card = dbc.Card(
    [
        dbc.CardHeader("Document Length Bubble Chart", className="guardian-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-bubble-chart",
                type="circle",
                children=[dcc.Graph(id='bubble-chart', style={"height": "600px"})]
            ),
            className="guardian-card-body"
        )
    ],
    className="mb-3 guardian-card",
)

bigrams_trigrams_card = dbc.Card(
    [
        dbc.CardHeader("Bigrams & Trigrams (Radar Chart)", className="guardian-header"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-bigrams-trigrams",
                type="circle",
                children=[dcc.Graph(id='bigrams-trigrams', style={"height": "600px"})]
            ),
            className="guardian-card-body"
        )
    ],
    className="mb-3 guardian-card",
)

article_table_card = dbc.Card(
    [
        dbc.CardHeader("Article Details", className="guardian-header"),
        dbc.CardBody([
            dash_table.DataTable(
                id='article-details',
                columns=[
                    {'name': 'Title', 'id': 'title', 'width': '50%'},
                    {'name': 'Published', 'id': 'published', 'width': '15%'},
                    {'name': 'Topics', 'id': 'topics', 'width': '35%', 'presentation': 'markdown'},
                ],
                style_table={'overflowX': 'auto', 'border': '1px solid #ddd'},
                style_cell={
                    'textAlign': 'left', 
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'padding': '10px',
                    'fontFamily': 'Guardian Egyptian Web, Georgia, serif'
                },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#005689',
                    'color': 'white',
                    'padding': '12px 10px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f6f6f6'
                    }
                ],
                page_size=10,
                markdown_options={'html': True}
            )
        ], className="guardian-card-body")
    ],
    className="mb-3 guardian-card",
)

app.layout = dbc.Container(
    [
        guardian_theme_css,
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
    className="guardian-container"
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
    ]
)
def update_visuals(start_date, end_date, num_topics, perplexity):
    """
    1) Train LDA on the entire set within the selected date range.
    2) Build visuals & article table from the entire set.
    3) Create 3D t-SNE with user-chosen perplexity.
    4) Show bubble chart, etc.
    """
    try:
        logger.info(f"update_visuals: {start_date} to {end_date}, num_topics={num_topics}, perplexity={perplexity}")

        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date, num_topics)
        if df is None or df.empty:
            # Return empty figs
            fig_empty = go.Figure().update_layout(**get_guardian_plot_layout("No Data"))
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

        # Topic Word Dist
        words_list = []
        for t_id in range(num_topics):
            if 0 <= t_id < lda_model.num_topics:
                top_pairs = lda_model.show_topic(t_id, topn=10)
                for (w, prob) in top_pairs:
                    words_list.append((w, prob, t_id))

        if not words_list:
            fig_dist = go.Figure()
            fig_dist.update_layout(**get_guardian_plot_layout("No topics found"))
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
            fig_dist.update_layout(**get_guardian_plot_layout())

        # Word Cloud
        fig_wc = go.Figure()
        if num_topics > 0 and lda_model.num_topics > 0:
            fig_wc = create_word_cloud(lda_model.show_topic(0, topn=30))
        else:
            fig_wc.update_layout(**get_guardian_plot_layout("Word Cloud N/A"))

        # 3D t-SNE
        fig_tsne = create_tsne_visualization_3d(df, corpus, lda_model, perplexity)

        # Bubble Chart
        fig_bubble = create_bubble_chart(df)

        # Bigrams & Trigrams Radar
        fig_ngram = create_ngram_radar_chart(texts)

        # Article Table
        table_data = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            # Use <br> tags instead of \n for proper rendering in markdown
            these_topics = [
                f"**Topic {tid}**: {w:.3f}" for (tid, w)
                in sorted(doc_topics, key=lambda x: x[1], reverse=True)
            ]
            table_data.append({
                'title': df.at[i, 'title'],
                'published': df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '<br>'.join(these_topics)
            })

        return fig_dist, fig_wc, fig_tsne, fig_bubble, fig_ngram, table_data

    except Exception as e:
        logger.error(f"update_visuals error: {e}", exc_info=True)
        # Show error fig
        fig_err = go.Figure()
        fig_err.update_layout(**get_guardian_plot_layout(f"Error: {e}"))
        return fig_err, fig_err, fig_err, fig_err, fig_err, []

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
