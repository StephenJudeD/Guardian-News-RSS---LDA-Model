import dash
from dash import Dash, dcc, html, Input, Output, dash_table
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
# Expanded Stop Words & Global Colouring Variables
# ─────────────────────────────────────────────────────────────────────
CUSTOM_STOP_WORDS = {
    'says', 'said', 'would', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'new', 'like', 'get', 'make', 'first', 'year', 'years', 'time', 'way', 'says', 'say', 'saying', 'according',
    'told', 'reuters', 'guardian', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'week', 'month', 'us', 'people', 'government', 'could', 'will', 'may', 'trump', 'published', 'article', 'editor',
    'nt', 'dont', 'doesnt', 'cant', 'couldnt', 'shouldnt'
}

# Define colours – using Guardian blue and a poppy purple accent.
NAVY_BLUE = "#052962"
POPPY_PURPLE = "#A020F0"
HEADER_BACKGROUND = f"linear-gradient(90deg, {NAVY_BLUE}, {POPPY_PURPLE})"
# Pre-assign colours for our 5 topics (keys must be integers)
TOPIC_COLORS = {
    0: POPPY_PURPLE,
    1: NAVY_BLUE,
    2: "#8A2BE2",  # Blue Violet
    3: "#00008B",  # Dark Blue
    4: "#FF69B4"   # Hot Pink
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
    Fetch Guardian articles from start_date to end_date,
    then tokenize, detect bigrams/trigrams, and train an LDA model.
    Returns: (df, texts, dictionary, corpus, lda_model)
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
            filtered = [w.lower() for w in words if w.isalnum() and w.lower() not in stop_words]
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
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────
def create_word_cloud(topic_words):
    """
    Create a word cloud from LDA topic-word pairs.
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

def create_tsne_visualization_2d(df, corpus, lda_model):
    """
    Create a 2D t-SNE scatter plot of document topics.
    """
    try:
        if df is None or len(df) < 2:
            return go.Figure().update_layout(template='plotly', title='Not enough documents for t-SNE')

        doc_topics_list = []
        for i in df.index:
            topic_weights = [0.0] * lda_model.num_topics
            for topic_id, w in lda_model[corpus[i]]:
                topic_weights[topic_id] = w
            doc_topics_list.append(topic_weights)

        doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
        if len(doc_topics_array) < 2:
            return go.Figure().update_layout(template='plotly', title='Not enough docs for t-SNE')

        # Configure perplexity based on sample size
        perplex_val = 30 if len(doc_topics_array) >= 30 else max(2, len(doc_topics_array)-1)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplex_val, n_jobs=1)
        embedded = tsne.fit_transform(doc_topics_array)  # shape: (N, 2)

        scatter_df = pd.DataFrame({
            'x': embedded[:, 0],
            'y': embedded[:, 1],
            'dominant_topic': [np.argmax(row) for row in doc_topics_array],
            'title': df['title']
        })

        fig = px.scatter(
            scatter_df,
            x='x',
            y='y',
            color='dominant_topic',
            color_discrete_map=TOPIC_COLORS,
            hover_data=['title'],
            title='2D t-SNE Topic Clustering'
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating 2D t-SNE: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=str(e))

def create_bubble_chart(df):
    """
    Creates a bubble chart of published date vs. capped document length with an animation slider.
    Document lengths are capped at 1000 tokens.
    """
    try:
        if df is None or df.empty:
            return go.Figure().update_layout(template='plotly', title='Bubble Chart Unavailable')
        
        # Cap document length at 1000
        df['capped_doc_length'] = df['doc_length'].apply(lambda x: min(x, 1000))
        # Create a string version of the published date for the animation slider
        df['published_str'] = df['published'].dt.strftime('%Y-%m-%d')
        
        fig = px.scatter(
            df,
            x='published',
            y='capped_doc_length',
            size='capped_doc_length',
            color='dominant_topic',
            animation_frame='published_str',
            color_discrete_map=TOPIC_COLORS,
            hover_data=['title'],
            title='Document Length Bubble Chart (Max 1000)'
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=str(e))

def create_ngram_bar_chart(texts):
    """
    Create a bar chart of the most common bigrams/trigrams (tokens containing underscores).
    """
    try:
        ngram_counts = {}
        for tokens in texts:
            for tok in tokens:
                if "_" in tok:
                    ngram_counts[tok] = ngram_counts.get(tok, 0) + 1

        if not ngram_counts:
            return go.Figure().update_layout(template='plotly', title="No bigrams/trigrams found")

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
        fig.update_layout(template='plotly', yaxis={'categoryorder': 'total ascending'})
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram bar chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=str(e))

def create_topics_over_time(filtered_df):
    """
    Create a stacked bar chart showing the number of articles per dominant topic per day.
    """
    try:
        if filtered_df is None or filtered_df.empty:
            return go.Figure().update_layout(template='plotly', title="No Topics Over Time Data")
        
        df_time = filtered_df.copy()
        df_time['publish_date'] = df_time['published'].dt.date
        grouped = df_time.groupby(['publish_date', 'dominant_topic']).size().reset_index(name='count')
        pivot_df = grouped.pivot(index='publish_date', columns='dominant_topic', values='count').fillna(0)
        pivot_df = pivot_df.sort_index()

        fig = go.Figure()
        for topic in pivot_df.columns:
            fig.add_trace(go.Bar(
                x=pivot_df.index, 
                y=pivot_df[topic],
                name=f"Topic {topic}",
                marker_color=TOPIC_COLORS.get(topic, POPPY_PURPLE)
            ))
        fig.update_layout(
            barmode='stack',
            title="Topics Over Time",
            xaxis_title="Publish Date",
            yaxis_title="Article Count",
            template="plotly"
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating topics over time: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=f"Error: {e}")

def create_topic_donut_chart(filtered_df):
    """
    Create a donut chart that summarizes the counts of dominant topics in the filtered data.
    """
    try:
        if filtered_df is None or filtered_df.empty:
            return go.Figure().update_layout(template='plotly', title="No Data for Donut Chart")
        
        topic_counts = filtered_df['dominant_topic'].value_counts().sort_index()
        labels = [f"Topic {i}" for i in topic_counts.index]
        values = topic_counts.values
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker=dict(colors=[TOPIC_COLORS.get(i, POPPY_PURPLE) for i in topic_counts.index])
        )])
        fig.update_layout(
            title="Topic Distribution Donut Chart",
            template="plotly"
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating topic donut chart: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────
# Navbar & Cards (Layout)
# ─────────────────────────────────────────────────────────────────────
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
            className="g-0"
        )
    ],
    color=NAVY_BLUE,
    dark=True,
    className="mb-2 px-3"
)

explainer_card = dbc.Card(
    [
        dbc.CardHeader("About This App", style={"background": HEADER_BACKGROUND, "color": "white"}),
        dbc.CardBody(
            [
                html.P(
                    [
                        "This dashboard fetches articles from the Guardian’s RSS, processes them with Natural Language Processing (NLP), "
                        "applies LDA for topic modeling (with bigrams/trigrams detection), and uses techniques like 2D t-SNE for clustering. "
                        "Use the dynamic slider to adjust the time window (in days ago) and filter topics to see how news stories shift over time. ",
                        html.A(
                            "Code & Readme available on GitHub",
                            href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model/tree/main",
                            target="_blank",
                            style={"color": "blue", "textDecoration": "underline"}
                        )
                    ],
                    className="mb-0"
                )
            ],
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

# Use a dynamic range slider for the time filter (in days ago)
time_slider_card = dbc.Col(
    dbc.Card(
        [
            dbc.CardHeader("Select Time Range (Days Ago)", style={"background": HEADER_BACKGROUND, "color": "white"}),
            dbc.CardBody(
                [
                    dcc.RangeSlider(
                        id='days-slider',
                        min=0,
                        max=30,  # Maximum days ago; adjust as needed or dynamically
                        step=1,
                        value=[0, 14],  # Default: from today (0 days ago) to 14 days ago
                        marks={i: {'label': f"{i}d"} for i in range(0, 31, 5)},
                        tooltip={"placement": "bottom"}
                    )
                ]
            )
        ],
        className="mb-2",
        style={"backgroundColor": "white"}
    ),
    md=4
)

# Dropdown for topic filtering remains unchanged
topic_filter_card = dbc.Col(
    dbc.Card(
        [
            dbc.CardHeader("Select Topics", style={"background": HEADER_BACKGROUND, "color": "white"}),
            dbc.CardBody(
                [
                    dcc.Dropdown(
                        id='topic-filter',
                        options=[{'label': f'Topic {i}', 'value': i} for i in range(5)],
                        multi=True,
                        placeholder="Filter by topics...",
                        className="mb-2"
                    )
                ]
            )
        ],
        className="mb-2",
        style={"backgroundColor": "white"}
    ),
    md=4
)

# Empty column for spacing (if needed)
empty_col = dbc.Col(md=4)

controls_row = dbc.Row([time_slider_card, topic_filter_card, empty_col], className="my-2 px-2")

# Cards for Visualizations
topic_dist_card = dbc.Card(
    [
        dbc.CardHeader("Topic Word Distributions", style={"background": HEADER_BACKGROUND, "color": "white", "fontWeight": "bold"}),
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

tsne_2d_card = dbc.Card(
    [
        dbc.CardHeader("2D t-SNE Topic Clustering", style={"background": HEADER_BACKGROUND, "color": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Loading(
                id="loading-2d-tsne",
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
        dbc.CardHeader("Word Cloud", style={"background": HEADER_BACKGROUND, "color": "white", "fontWeight": "bold"}),
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
        dbc.CardHeader("Document Length Bubble Chart", style={"background": HEADER_BACKGROUND, "color": "white", "fontWeight": "bold"}),
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

topics_over_time_card = dbc.Card(
    [
        dbc.CardHeader("Topics Over Time", style={"background": HEADER_BACKGROUND, "color": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Loading(
                id="loading-topics-over-time",
                type="circle",
                children=[dcc.Graph(id='topics-over-time', style={"height": "600px"})]
            ),
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

topic_donut_card = dbc.Card(
    [
        dbc.CardHeader("Topic Distribution Donut", style={"background": HEADER_BACKGROUND, "color": "white", "fontWeight": "bold"}),
        dbc.CardBody(
            dcc.Loading(
                id="loading-topic-donut",
                type="circle",
                children=[dcc.Graph(id='topic-donut', style={"height": "600px"})]
            ),
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

bigrams_trigrams_card = dbc.Card(
    [
        dbc.CardHeader("Bigrams & Trigrams", style={"background": HEADER_BACKGROUND, "color": "white", "fontWeight": "bold"}),
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
        dbc.CardHeader("Article Details", style={"background": HEADER_BACKGROUND, "color": "white"}),
        dbc.CardBody(
            [
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
                        'backgroundColor': HEADER_BACKGROUND,
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    page_size=10
                )
            ],
            style={"backgroundColor": "white"}
        )
    ],
    className="mb-3",
    style={"backgroundColor": "white"}
)

app.layout = dbc.Container([
    navbar,
    dbc.Row([dbc.Col(explainer_card, md=12)], className="g-3"),
    controls_row,
    dbc.Row([dbc.Col(topic_dist_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(tsne_2d_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(wordcloud_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(bubble_chart_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(topics_over_time_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(topic_donut_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(bigrams_trigrams_card, md=12)], className="g-3"),
    dbc.Row([dbc.Col(article_table_card, md=12)], className="g-3"),
], fluid=True, style={"backgroundColor": "#f9f9f9"})

# ─────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────
@app.callback(
    [
        Output('topic-distribution', 'figure'),
        Output('word-cloud', 'figure'),
        Output('tsne-plot', 'figure'),
        Output('bubble-chart', 'figure'),
        Output('bigrams-trigrams', 'figure'),
        Output('topics-over-time', 'figure'),
        Output('topic-donut', 'figure'),
        Output('article-details', 'data')
    ],
    [
        Input('days-slider', 'value'),
        Input('topic-filter', 'value')
    ]
)
def update_visuals(days_range, selected_topics):
    try:
        # Convert slider values (days ago) to actual dates.
        # days_range = [min_val, max_val] where min_val is the newer date (0 = today)
        today = datetime.now().date()
        start_date = (today - timedelta(days=days_range[1])).strftime('%Y-%m-%d')
        end_date = (today - timedelta(days=days_range[0])).strftime('%Y-%m-%d')

        logger.info(f"update_visuals: {start_date} to {end_date}, topics={selected_topics}")
        
        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date)
        if df is None or df.empty:
            empty_fig = go.Figure().update_layout(template='plotly', title="No Data")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

        # If no topic is selected, default to all topics (0 through 4)
        if not selected_topics:
            selected_topics = list(range(lda_model.num_topics))

        # Build document-level info: document length and dominant topic
        doc_lengths = []
        doc_dominant_topics = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            n_tokens = len(texts[i] if texts[i] else [])
            doc_lengths.append(n_tokens)
            best_t = max(doc_topics, key=lambda x: x[1])[0] if doc_topics else -1
            doc_dominant_topics.append(best_t)
        df["doc_length"] = doc_lengths
        df["dominant_topic"] = doc_dominant_topics

        # Filter articles with dominant topics in selected_topics
        filtered_df = df[df["dominant_topic"].isin(selected_topics)].copy()
        if filtered_df.empty:
            empty_fig = go.Figure().update_layout(template='plotly', title="No Data (Dominant Topic Filter)")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []
        
        filtered_texts = [texts[i] for i in filtered_df.index]
        filtered_corpus = [corpus[i] for i in filtered_df.index]

        # 1) Topic Word Distribution
        words_list = []
        for t_id in selected_topics:
            top_pairs = lda_model.show_topic(t_id, topn=20)
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
                color_discrete_map=TOPIC_COLORS,
                orientation="h",
                title="Topic Word Distributions"
            )
            dist_fig.update_layout(template='plotly', yaxis={'categoryorder': 'total ascending'})

        # 2) Word Cloud for the first selected topic
        first_topic = selected_topics[0]
        wc_fig = create_word_cloud(lda_model.show_topic(first_topic, topn=30))

        # 3) 2D t-SNE visualization
        tsne_fig = create_tsne_visualization_2d(df, corpus, lda_model)

        # 4) Bubble Chart (with animation slider from published_str)
        bubble_fig = create_bubble_chart(filtered_df)

        # 5) N-grams (bigrams/trigrams) Bar Chart
        ngram_fig = create_ngram_bar_chart(filtered_texts)

        # 6) Topics Over Time (stacked bar chart)
        topics_over_time_fig = create_topics_over_time(filtered_df)

        # 7) Topic Distribution Donut Chart
        donut_fig = create_topic_donut_chart(filtered_df)

        # 8) Article Details Table Data
        table_data = []
        for i in filtered_df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            these_topics = [
                f"Topic {tid}: {w:.3f}" for (tid, w) in sorted(doc_topics, key=lambda x: x[1], reverse=True)
                if tid in selected_topics
            ]
            table_data.append({
                'title': filtered_df.at[i, 'title'],
                'published': filtered_df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(these_topics)
            })

        return dist_fig, wc_fig, tsne_fig, bubble_fig, ngram_fig, topics_over_time_fig, donut_fig, table_data

    except Exception as e:
        logger.error(f"update_visuals error: {e}", exc_info=True)
        empty_fig = go.Figure().update_layout(template='plotly', title=f"Error: {e}")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
