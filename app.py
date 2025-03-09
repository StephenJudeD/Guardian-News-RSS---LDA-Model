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

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Expanded Stop Words
CUSTOM_STOP_WORDS = {
    'says', 'said', 'would', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'new', 'like', 'get', 'make', 'first', 'year', 'years', 'time', 'way', 'says', 'say', 'saying', 'according',
    'told', 'reuters', 'guardian', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'week', 'month', 'us', 'people', 'government', 'could', 'will', 'may', 'trump', 'published', 'article', 'editor',
    'nt', 'dont', 'doesnt', 'cant', 'couldnt', 'shouldnt', 'last', 'well', 'still', 'price',
    'breaking', 'update', 'live', 'say'
}

# Environment variables & NLTK
load_dotenv()
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
if not GUARDIAN_API_KEY:
    logger.error("⚠️ No GUARDIAN_API_KEY found in environment!")
    raise ValueError("No GUARDIAN_API_KEY found in environment!")

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(CUSTOM_STOP_WORDS)

# GuardianFetcher
guardian = GuardianFetcher(GUARDIAN_API_KEY)

# Dash Setup - Using built-in dark theme
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server
app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def process_articles(start_date, end_date, num_topics=3):
    """
    Fetch Guardian articles in the given date range,
    then tokenize, detect bigrams/trigrams, and train LDA on the entire set.
    Returns (df, texts, dictionary, corpus, lda_model, coherence).
    """
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date} with num_topics={num_topics}")

        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        days_back = (datetime.now().date() - start_date_dt).days + 1

        df = guardian.fetch_articles(days_back=days_back, page_size=100)
        if df.empty:
            logger.warning("No articles fetched!")
            return None, None, None, None, None, None

        df = df[
            (df['published'].dt.date >= start_date_dt) &
            (df['published'].dt.date <= end_date_dt)
        ]
        logger.info(f"Filtered to {len(df)} articles in date range")
        if len(df) < 5:
            logger.warning("Not enough articles for LDA.")
            return None, None, None, None, None, None

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
                if w.isalnum() and len(w) > 2 and w.lower() not in stop_words
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
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        corpus = [dictionary.doc2bow(t) for t in texts]

        # Train LDA with more conservative settings
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=5,
            random_state=42,
            alpha='auto'
        )
        
        # Calculate simple topic coherence (average of top term probabilities)
        coherence = {}
        for topic_id in range(num_topics):
            top_terms = lda_model.show_topic(topic_id, topn=10)
            coherence[topic_id] = sum(prob for _, prob in top_terms) / len(top_terms)

        logger.info(f"Processed {len(df)} articles successfully with LDA num_topics={num_topics}")
        return df, texts, dictionary, corpus, lda_model, coherence

    except Exception as e:
        logger.error(f"Error in process_articles: {e}", exc_info=True)
        return None, None, None, None, None, None

# ─────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────
def create_word_cloud(topic_words):
    """
    Create a word cloud from LDA topic-word pairs for dark theme.
    """
    try:
        freq_dict = dict(topic_words)
        
        wc = WordCloud(
            background_color="#222",
            width=800,
            height=400,
            colormap="viridis",
            max_words=50,
            prefer_horizontal=0.9,
            contour_width=1,
            contour_color='#444'
        ).generate_from_frequencies(freq_dict)

        fig = px.imshow(wc)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(template="plotly_dark")
        return fig

def create_tsne_visualization_3d(df, corpus, lda_model, perplexity=15):
    """
    3D t-SNE scatter (Plotly).
    Optimized for dark theme & performance.
    """
    try:
        if df is None or len(df) < 2:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark")
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
            fig.update_layout(template="plotly_dark")
            return fig

        # Use simpler t-SNE settings
        perplex_val = min(perplexity, max(2, len(doc_topics_array) // 3))
        tsne = TSNE(
            n_components=3,
            random_state=42,
            perplexity=perplex_val,
            n_jobs=1,
            n_iter=250  # Reduce iterations for performance
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
            title=f'3D Topic Clusters (Perplexity={perplex_val})'
        )
        
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title='')
            ),
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating 3D t-SNE: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(template="plotly_dark")
        return fig

def create_bubble_chart(df, selected_topic=None):
    """
    Bubble chart: doc length vs published date, sized by doc length,
    colored by dominant_topic. Optimized for dark theme.
    Can filter by selected topic.
    """
    try:
        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark")
            return fig
            
        # Filter by selected topic if specified
        if selected_topic is not None and selected_topic != 'all':
            filtered_df = df[df['dominant_topic'] == int(selected_topic)].copy()
            if filtered_df.empty:
                fig = go.Figure()
                fig.update_layout(template="plotly_dark", title="No articles for selected topic")
                return fig
        else:
            filtered_df = df.copy()

        # Remove outliers
        cut_off = filtered_df['doc_length'].quantile(0.95)
        filtered_df = filtered_df[filtered_df['doc_length'] <= cut_off]
        
        if filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark")
            return fig

        fig = px.scatter(
            filtered_df,
            x='published',
            y='doc_length',
            size='doc_length',
            color='dominant_topic',
            size_max=25,
            hover_data=['title'],
            title='Document Length Over Time',
            log_y=True
        )
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Publication Date",
            yaxis_title="Document Length (words)",
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(template="plotly_dark")
        return fig

def create_ngram_radar_chart(texts):
    """
    Radar (sonar) chart of the most common bigrams/trigrams (top 8).
    Optimized for dark theme.
    """
    try:
        ngram_counts = {}
        for tokens in texts:
            for tok in tokens:
                if "_" in tok:
                    ngram_counts[tok] = ngram_counts.get(tok, 0) + 1

        if not ngram_counts:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark", title="No multi-word phrases found")
            return fig

        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        top_ngrams = sorted_ngrams[:8]  # Take fewer for better visibility
        
        # Format the ngrams for better display
        formatted_ngrams = []
        for ngram, count in top_ngrams:
            formatted = ngram.replace('_', ' ')
            formatted_ngrams.append((formatted, count))
            
        df_ngram = pd.DataFrame(formatted_ngrams, columns=["ngram", "count"])

        fig = px.line_polar(
            df_ngram,
            r="count",
            theta="ngram",
            line_close=True,
            title="Top Phrases (Radar)"
        )
        
        # Enhance the polar chart for dark theme
        fig.update_traces(
            fill='toself',
            fillcolor="rgba(65, 105, 225, 0.3)",
            line=dict(color="royalblue", width=2)
        )
        
        fig.update_layout(
            template="plotly_dark",
            polar=dict(
                radialaxis=dict(visible=True, color="#888"),
                angularaxis=dict(color="#888", tickfont=dict(size=11))
            ),
            showlegend=False
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram radar chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(template="plotly_dark")
        return fig

def create_topic_distribution(topic_distributions, selected_topic=None):
    """
    Create horizontal bar chart of word distributions in topics.
    Can highlight a selected topic.
    """
    try:
        words_list = []
        
        # Filter by selected topic if specified
        if selected_topic is not None and selected_topic != 'all':
            if int(selected_topic) in topic_distributions:
                for word, prob in topic_distributions[int(selected_topic)]:
                    words_list.append((word, prob, int(selected_topic)))
        else:
            # Use all topics
            for topic_id, words in topic_distributions.items():
                for word, prob in words:
                    words_list.append((word, prob, topic_id))
        
        if not words_list:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark", title="No topics available")
            return fig
            
        df_dist = pd.DataFrame(words_list, columns=["word", "prob", "topic"])
        
        fig = px.bar(
            df_dist,
            x="prob",
            y="word",
            color="topic",
            orientation="h",
            title="Topic Keywords"
        )
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Weight",
            yaxis_title="",
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating topic distribution: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(template="plotly_dark")
        return fig

def create_coherence_chart(coherence):
    """
    Create a simple bar chart showing topic coherence
    """
    try:
        if not coherence:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark", title="No coherence data available")
            return fig
            
        topics = list(coherence.keys())
        values = list(coherence.values())
        
        fig = px.bar(
            x=topics, 
            y=values,
            labels={'x': 'Topic', 'y': 'Coherence Score'},
            title="Topic Coherence Scores"
        )
        
        # Add value labels on bars
        fig.update_traces(
            text=[f'{v:.3f}' for v in values],
            textposition='outside'
        )
        
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(tickmode='array', tickvals=topics, ticktext=[f'Topic {t}' for t in topics])
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating coherence chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(template="plotly_dark")
        return fig

# ─────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="https://static.guim.co.uk/sys-images/Guardian/Pix/pictures/2010/03/01/poweredbyguardianBLACK.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("Guardian News Topic Explorer", className="ms-2")),
                    ],
                    align="center"
                ),
                href="#",
            ),
        ],
        fluid=True
    ),
    color="primary",
    dark=True,
)

about_card = dbc.Card(
    [
        dbc.CardHeader("About This Dashboard"),
        dbc.CardBody(
            [
                html.P(
                    [
                        "This dashboard processes Guardian news articles with LDA topic modeling to discover hidden themes. "
                        "Explore articles by date range, topic selection, and examine their distribution in 3D space. "
                        "Use the controls below to adjust parameters and see how topics emerge across time.",
                        html.A(
                            " Source on GitHub",
                            href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model",
                            target="_blank",
                            className="ms-1"
                        ),
                    ],
                ),
            ]
        ),
    ],
    className="mb-3",
)

date_filter_card = dbc.Card(
    [
        dbc.CardHeader("Date Range"),
        dbc.CardBody([
            dbc.ButtonGroup(
                [
                    dbc.Button("Last Day", id="date-1d", color="secondary", outline=True, className="me-1"),
                    dbc.Button("Last 3 Days", id="date-3d", color="secondary", outline=True, className="me-1"),
                    dbc.Button("Last Week", id="date-7d", color="secondary", outline=False, className="me-1"),
                ],
                className="mb-3 w-100"
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Start Date"),
                    dcc.DatePickerSingle(
                        id='start-date',
                        date=(datetime.now() - timedelta(days=7)).date(),
                        display_format='YYYY-MM-DD',
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("End Date"),
                    dcc.DatePickerSingle(
                        id='end-date',
                        date=datetime.now().date(),
                        display_format='YYYY-MM-DD',
                    ),
                ], width=6),
            ]),
        ]),
    ],
    className="mb-3",
)

topics_card = dbc.Card(
    [
        dbc.CardHeader("Topic Settings"),
        dbc.CardBody([
            dbc.Label("Number of Topics"),
            dcc.Slider(
                id="num-topics-slider",
                min=2,
                max=7,  # Reduced max as requested
                value=3,  # Default to 3 as requested
                step=1,
                marks={i: str(i) for i in range(2, 8)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="mb-4"
            ),
            
            dbc.Label("Topic Selection"),
            dcc.Dropdown(
                id="topic-selector",
                options=[{"label": "All Topics", "value": "all"}],
                value="all",
                clearable=False,
            ),
        ]),
    ],
    className="mb-3",
)

tsne_card = dbc.Card(
    [
        dbc.CardHeader("t-SNE Settings"),
        dbc.CardBody([
            dbc.Label("3D Perplexity"),
            dcc.Slider(
                id="tsne-perplexity-slider",
                min=5,
                max=30,  # Reduced maximum value
                step=5,
                value=15,  # Reduced default value
                marks={i: str(i) for i in range(5, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ]),
    ],
    className="mb-3",
)

update_button = dbc.Button(
    "Update Analysis", 
    id="update-button", 
    color="primary", 
    size="lg", 
    className="w-100 mb-3"
)

controls_row = dbc.Row(
    [
        dbc.Col(date_filter_card, md=6, lg=4),
        dbc.Col([
            topics_card,
            update_button
        ], md=6, lg=4),
        dbc.Col(tsne_card, md=12, lg=4),
    ],
    className="mb-4"
)

# Main visualization cards
topic_dist_card = dbc.Card(
    [
        dbc.CardHeader("Topic Word Distributions"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-topic-dist",
                type="default",
                children=[dcc.Graph(id='topic-distribution', style={"height": "400px"})]
            )
        )
    ],
    className="mb-3",
)

coherence_card = dbc.Card(
    [
        dbc.CardHeader("Topic Coherence Scores"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-coherence",
                type="default",
                children=[dcc.Graph(id='coherence-chart', style={"height": "300px"})]
            )
        )
    ],
    className="mb-3",
)

wordcloud_card = dbc.Card(
    [
        dbc.CardHeader("Word Cloud"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-wordcloud",
                type="default",
                children=[dcc.Graph(id='word-cloud', style={"height": "400px"})]
            )
        )
    ],
    className="mb-3",
)

tsne_3d_card = dbc.Card(
    [
        dbc.CardHeader("3D Topic Clustering"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-3d-tsne",
                type="default",
                children=[dcc.Graph(id='tsne-plot', style={"height": "500px"})]
            )
        )
    ],
    className="mb-3",
)

bubble_chart_card = dbc.Card(
    [
        dbc.CardHeader("Document Length Over Time"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-bubble-chart",
                type="default",
                children=[dcc.Graph(id='bubble-chart', style={"height": "400px"})]
            )
        )
    ],
    className="mb-3",
)

ngram_chart_card = dbc.Card(
    [
        dbc.CardHeader("Common Phrases (Radar)"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-ngram-chart",
                type="default",
                children=[dcc.Graph(id='ngram-chart', style={"height": "400px"})]
            )
        )
    ],
    className="mb-3",
)

article_table_card = dbc.Card(
    [
        dbc.CardHeader("Article Details"),
        dbc.CardBody(
            dcc.Loading(
                id="loading-article-table",
                type="default",
                children=[
                    html.Div(id="article-count", className="mb-3"),
                    dash_table.DataTable(
                        id='article-table',
                        columns=[
                            {'name': 'Title', 'id': 'title', 'presentation': 'markdown'},
                            {'name': 'Published', 'id': 'published'},
                            {'name': 'Topics', 'id': 'topics', 'presentation': 'markdown'},
                        ],
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'padding': '10px',
                            'fontSize': '14px'
                        },
                        style_header={
                            'fontWeight': 'bold',
                            'backgroundColor': '#333',
                            'color': 'white',
                        },
                        style_data={
                            'backgroundColor': '#222',
                            'color': 'white',
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#333',
                            }
                        ],
                        page_size=10,
                        markdown_options={'html': True}
                    )
                ]
            )
        )
    ],
    className="mb-3",
)

app.layout = dbc.Container(
    [
        navbar,
        dbc.Row([dbc.Col(about_card)], className="mt-3"),
        controls_row,
        
        # Main visualizations
        dbc.Row([
            dbc.Col(topic_dist_card, md=8),
            dbc.Col(coherence_card, md=4),
        ]),
        
        dbc.Row([
            dbc.Col(wordcloud_card, md=6),
            dbc.Col(ngram_chart_card, md=6),
        ]),
        
        dbc.Row([dbc.Col(tsne_3d_card)]),
        
        dbc.Row([dbc.Col(bubble_chart_card)]),
        
        dbc.Row([dbc.Col(article_table_card)]),
        
        html.Footer(
            html.P(
                "Data from The Guardian, via their public API. This is a non-commercial educational project.",
                className="text-center text-muted mt-4 mb-4"
            )
        ),
    ],
    fluid=True,
    className="pb-5"
)

# ─────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────

# Date range button callbacks
@app.callback(
    [Output("start-date", "date"), Output("end-date", "date"),
     Output("date-1d", "color"), Output("date-3d", "color"), Output("date-7d", "color")],
    [Input("date-1d", "n_clicks"), Input("date-3d", "n_clicks"), Input("date-7d", "n_clicks")],
    [State("date-1d", "n_clicks"), State("date-3d", "n_clicks"), State("date-7d", "n_clicks")]
)
def update_date_range(n1, n3, n7, s1, s3, s7):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Default state
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        return start_date, end_date, "secondary", "secondary", "primary"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    end_date = datetime.now().date()
    
    if button_id == "date-1d":
        start_date = end_date - timedelta(days=1)
        return start_date, end_date, "primary", "secondary", "secondary"
    elif button_id == "date-3d":
        start_date = end_date - timedelta(days=3)
        return start_date, end_date, "secondary", "primary", "secondary"
    else:  # date-7d
        start_date = end_date - timedelta(days=7)
        return start_date, end_date, "secondary", "secondary", "primary"

# Update topic selector options
@app.callback(
    Output("topic-selector", "options"),
    [Input("num-topics-slider", "value")]
)
def update_topic_options(num_topics):
    options = [{"label": "All Topics", "value": "all"}]
    for i in range(num_topics):
        options.append({"label": f"Topic {i}", "value": str(i)})
    return options

# Main visualization callback
@app.callback(
    [
        Output("topic-distribution", "figure"),
        Output("coherence-chart", "figure"),
        Output("word-cloud", "figure"),
        Output("tsne-plot", "figure"),
        Output("bubble-chart", "figure"),
        Output("ngram-chart", "figure"),
        Output("article-table", "data"),
        Output("article-count", "children")
    ],
    [
        Input("update-button", "n_clicks"),
        Input("topic-selector", "value")
    ],
    [
        State("start-date", "date"),
        State("end-date", "date"),
        State("num-topics-slider", "value"),
        State("tsne-perplexity-slider", "value")
    ]
)
def update_visuals(n_clicks, selected_topic, start_date, end_date, num_topics, perplexity):
    """
    Update all visualizations based on input parameters and selected topic.
    """
    if n_clicks is None:  # Initial load
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], "No data yet. Click 'Update Analysis' to begin."
    
    try:
        logger.info(f"update_visuals: {start_date} to {end_date}, num_topics={num_topics}, perplexity={perplexity}")

        # Process articles
        df, texts, dictionary, corpus, lda_model, coherence = process_articles(start_date, end_date, num_topics)
        
        if df is None or df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                template="plotly_dark",
                title="No articles found in the selected date range"
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], "No articles found in the selected date range."

        # Extract document lengths and dominant topics
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

        # Extract topic term distributions
        topic_distributions = {}
        for t_id in range(num_topics):
            topic_distributions[t_id] = lda_model.show_topic(t_id, topn=10)

        # Topic Distribution
        fig_dist = create_topic_distribution(topic_distributions, selected_topic)
        
        # Coherence
        fig_coherence = create_coherence_chart(coherence)
        
        # Word Cloud - use selected topic if specified
        fig_wc = go.Figure()
        if selected_topic != 'all' and int(selected_topic) < num_topics:
            fig_wc = create_word_cloud(lda_model.show_topic(int(selected_topic), topn=30))
        else:
            fig_wc = create_word_cloud(lda_model.show_topic(0, topn=30))

        # 3D t-SNE
        fig_tsne = create_tsne_visualization_3d(df, corpus, lda_model, perplexity)

        # Bubble Chart
        fig_bubble = create_bubble_chart(df, selected_topic)

        # Ngram Radar
        fig_ngram = create_ngram_radar_chart(texts)

        # Article table data
        table_data = []
        filtered_df = df
        
        # Filter by selected topic if necessary
        if selected_topic != 'all':
            filtered_df = df[df['dominant_topic'] == int(selected_topic)].copy()
            
        for i in filtered_df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            
            # Use <br> for markdown in the table
            these_topics = [
                f"**Topic {tid}**: {w:.3f}" for (tid, w)
                in sorted(doc_topics, key=lambda x: x[1], reverse=True)
            ]
            
            # Make title clickable and open in new tab
            title_with_link = f"[{filtered_df.at[i, 'title']}](https://www.theguardian.com/search?q={filtered_df.at[i, 'title'].replace(' ', '+')}) <i class='fas fa-external-link-alt' style='font-size: 0.8em'></i>"
            
            table_data.append({
                'title': title_with_link,
                'published': filtered_df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': "<br>".join(these_topics)
            })

        # Article count message
        count_message = html.Div([
            html.Strong(f"Found {len(filtered_df)} articles"),
            html.Span(f" from {start_date} to {end_date}"),
            html.Span(f" for Topic {selected_topic}" if selected_topic != 'all' else " across all topics")
        ])

        return fig_dist, fig_coherence, fig_wc, fig_tsne, fig_bubble, fig_ngram, table_data, count_message

    except Exception as e:
        logger.error(f"update_visuals error: {e}", exc_info=True)
        # Show error fig
        fig_err = go.Figure()
        fig_err.update_layout(
            template="plotly_dark",
            title=f"Error: {e}"
        )
        return fig_err, fig_err, fig_err, fig_err, fig_err, fig_err, [], f"Error: {e}"

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=False, port=port)
