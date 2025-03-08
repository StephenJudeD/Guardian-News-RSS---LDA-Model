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
# Dash Setup
# ─────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

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
    Create a word cloud from LDA topic-word pairs, white background.
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

def create_tsne_visualization_3d(df, corpus, lda_model, perplexity=30):
    """
    3D t-SNE scatter (Plotly).
    Uses all documents in df/corpus to avoid filtering by topic.
    Allows dynamic perplexity setting.
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

        # If we have very few documents, reduce perplexity accordingly
        perplex_val = min(perplexity, max(2, len(doc_topics_array) - 1))

        tsne = TSNE(
            n_components=3,
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
            title=f'3D t-SNE Topic Clustering (Perplexity={perplex_val})'
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating 3D t-SNE: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=str(e))

def create_bubble_chart(df):
    """
    Bubble chart: doc length vs published date, sized by doc length,
    colored by dominant_topic.
    """
    try:
        if df is None or df.empty:
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
        return go.Figure().update_layout(template='plotly', title=str(e))

# NEW/CHANGED: create_ngram_radar_chart instead of bar chart
def create_ngram_radar_chart(texts):
    """
    Radar (sonar) chart of the most common bigrams/trigrams.
    We'll pick the top 10 by frequency.
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

        # Sort by frequency and limit to top 10
        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        top_ngrams = sorted_ngrams[:10]
        df_ngram = pd.DataFrame(top_ngrams, columns=["ngram", "count"])

        # A radar chart in Plotly can be done with "line_polar" or "scatterpolar"
        fig = px.line_polar(
            df_ngram,
            r="count",
            theta="ngram",
            line_close=True,
            title="Top Bigrams & Trigrams (Radar Chart)"
        )
        # Fill the area for a more ‘sonar-like’ appearance
        fig.update_traces(fill='toself')
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram radar chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=str(e))

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
                        style={"color": "white", "fontWeight": "bold", "fontSize": "2rem"}
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

explainer_card = dbc.Card(
    [
        dbc.CardHeader("About This App", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
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
                            style={"color": "blue", "textDecoration": "underline"},
                        ),
                    ],
                    className="mb-0",
                )
            ],
            style={"backgroundColor": "white"},
        ),
    ],
    className="mb-3",
    style={"backgroundColor": "white"},
)

# NEW/CHANGED: Radio Items for Last Day, Last 3 Days, Last Week
date_filter_card = dbc.Card(
    [
        dbc.CardHeader("Select Date Range", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
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
                end_date=datetime.now().date(),
                className="mb-2"
            )
        ]),
    ],
    className="mb-2",
    style={"backgroundColor": "white"}
)

topic_filter_card = dbc.Card(
    [
        dbc.CardHeader("Select Topics (IDs)", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
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
)

lda_controls_card = dbc.Card(
    [
        dbc.CardHeader("Advanced LDA Controls", style={"backgroundColor": NAVY_BLUE, "color": "white"}),
        dbc.CardBody([
            html.Label("Number of Topics (2-15):", style={"fontWeight": "bold"}),
            dcc.Slider(
                id="num-topics-slider",
                min=2,
                max=15,
                value=5,  # default
                step=1,
                marks={i: str(i) for i in range(2, 16)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="mb-4"
            ),
            html.Label("t-SNE Perplexity (5-50):", style={"fontWeight": "bold"}),
            dcc.Slider(
                id='tsne-perplexity-slider',
                min=5,
                max=50,
                step=5,
                value=30,  # default
                marks={i: str(i) for i in range(5, 51, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    ],
    className="mb-2",
    style={"backgroundColor": "white"}
)

controls_row = dbc.Row(
    [
        dbc.Col(date_filter_card, md=4),
        dbc.Col(topic_filter_card, md=4),
        dbc.Col(lda_controls_card, md=4),
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

# NEW/CHANGED: "Radar Chart" text instead of "Bigrams & Trigrams"
bigrams_trigrams_card = dbc.Card(
    [
        dbc.CardHeader("Bigrams & Trigrams (Radar Chart)", style={"backgroundColor": "white", "fontWeight": "bold"}),
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
    dbc.Row([dbc.Col(explainer_card, md=12)], className="g-3"),
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
    # NEW/CHANGED: Radio logic for Last Day, Last Three Days, Last Week
    if selected_range == 'last_day':
        start_date = end_date - timedelta(days=1)
    elif selected_range == 'last_three':
        start_date = end_date - timedelta(days=3)
    elif selected_range == 'last_week':
        start_date = end_date - timedelta(days=7)
    else:
        start_date = end_date - timedelta(days=7)
    return start_date, end_date

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
        Input('num-topics-slider', 'value'),
        Input('tsne-perplexity-slider', 'value')
    ]
)
def update_visuals(start_date, end_date, selected_topics, num_topics, perplexity):
    """
    Main callback:
     1) Train LDA on the entire set within the selected date range using 'num_topics'.
     2) If no topic is manually selected, show [0..num_topics-1].
     3) Build visuals & article table from the filtered set.
     4) Create 3D t-SNE from the entire df/corpus, with an adjustable perplexity.
    """
    try:
        logger.info(f"update_visuals: {start_date} to {end_date}, topics={selected_topics}, num_topics={num_topics}, perplexity={perplexity}")

        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date, num_topics)
        if df is None or df.empty:
            empty_fig = go.Figure().update_layout(template='plotly', title="No Data")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

        # If no topics are selected, default to all possible [0..num_topics-1]
        if not selected_topics:
            selected_topics = list(range(num_topics))

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

        # Filter articles by the selected topics
        filtered_df = df[df["dominant_topic"].isin(selected_topics)].copy()
        if filtered_df.empty:
            empty_fig = go.Figure().update_layout(
                template='plotly',
                title="No Data (Dominant Topic Filter)"
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

        # Build subset for visuals that need only the filtered set
        filtered_texts = []
        filtered_corpus = []
        for i in filtered_df.index:
            filtered_texts.append(texts[i])
            filtered_corpus.append(corpus[i])

        # 1) Topic Word Distribution (limit top words to 10 for each topic)
        words_list = []
        for t_id in selected_topics:
            if 0 <= t_id < lda_model.num_topics:
                top_pairs = lda_model.show_topic(t_id, topn=10)  # NEW/CHANGED: top 10
                for (w, prob) in top_pairs:
                    words_list.append((w, prob, t_id))
        if not words_list:
            dist_fig = go.Figure().update_layout(template='plotly', title="No topics found")
        else:
            df_dist = pd.DataFrame(words_list, columns=["word", "prob", "topic"])
            dist_fig = px.bar(
                df_dist,
                x="prob",
                y="word",
                color="topic",
                orientation="h",
                title="Topic Word Distributions (Top 10)"
            )
            dist_fig.update_layout(template='plotly', yaxis={'categoryorder': 'total ascending'})

        # 2) Word Cloud (take the first selected topic)
        first_topic = selected_topics[0]
        if 0 <= first_topic < lda_model.num_topics:
            wc_fig = create_word_cloud(lda_model.show_topic(first_topic, topn=30))
        else:
            wc_fig = go.Figure().update_layout(template='plotly', title="Word Cloud N/A")

        # 3) 3D t-SNE (use entire df/corpus)
        tsne_fig = create_tsne_visualization_3d(df, corpus, lda_model, perplexity)

        # 4) Bubble Chart (use filtered_df/corpus)
        bubble_fig = create_bubble_chart(filtered_df)

        # 5) Radar chart for Bigrams & Trigrams (use filtered texts)
        ngram_fig = create_ngram_radar_chart(filtered_texts)

        # 6) Article Table data (only for the filtered set)
        table_data = []
        for i in filtered_df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            # Show all topics in descending probability, but only if they appear in 'selected_topics'
            these_topics = [
                f"Topic {tid}: {w:.3f}" for (tid, w) in sorted(doc_topics, key=lambda x: x[1], reverse=True)
                if tid in selected_topics
            ]
            table_data.append({
                'title': filtered_df.at[i, 'title'],
                'published': filtered_df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
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
