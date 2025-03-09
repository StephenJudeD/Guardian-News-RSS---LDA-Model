import dash
from dash import Dash, html, dcc, Input, Output, dash_table, State, callback_context
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
    level=logging.INFO,
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
    'breaking', 'update', 'live', 'say', 'going', 'think', 'know', 'just', 'now', 'even', 'taking', 'back'
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

# Dash Setup
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server
app.config.suppress_callback_exceptions = True

# Guardian Theme Colors
GUARDIAN_COLORS = {
    "blue": "#005689",
    "blue_light": "#00b2ff",
    "red": "#c70000",
    "yellow": "#ffbb00",
    "background": "#f6f6f6",
    "border": "#dcdcdc",
}

def get_plot_layout(fig_title=""):
    """Return a default layout for Guardian-themed figures."""
    return dict(
        paper_bgcolor="white",
        plot_bgcolor="#f6f6f6",
        font=dict(
            family="Georgia, serif", 
            size=16,
            color="#333333"
        ),
        title=dict(text=fig_title, font=dict(size=20)),
        margin=dict(l=40, r=40, t=50, b=40),
        colorway=[
            "#005689", "#c70000", "#ffbb00", "#00b2ff", "#90dcff", 
            "#ff5b5b", "#4bc6df", "#aad801", "#43853d", "#767676"
        ],
    )

# Data Processing
@lru_cache(maxsize=64)
def process_articles(start_date, end_date, num_topics=3):
    """
    Process and model Guardian articles.
    """
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date} with num_topics={num_topics}")

        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        days_back = (datetime.now().date() - start_date_dt).days + 1

        # Fetch articles
        df = guardian.fetch_articles(days_back=days_back, page_size=50)
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
        dictionary.filter_extremes(no_below=3, no_above=0.85)
        corpus = [dictionary.doc2bow(t) for t in texts]

        # Train LDA
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=3,
            iterations=30,
            alpha='auto',
            random_state=42,
            chunksize=100
        )

        logger.info(f"Processed {len(df)} articles successfully with LDA num_topics={num_topics}")
        return df, texts, dictionary, corpus, lda_model

    except Exception as e:
        logger.error(f"Error in process_articles: {e}", exc_info=True)
        return None, None, None, None, None

# Layout Components
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src="https://static.guim.co.uk/sys-images/Guardian/Pix/pictures/2010/03/01/poweredbyguardianBLACK.png", height="30px"), width="auto"),
                dbc.Col(dbc.NavbarBrand("Guardian News Topic Explorer")),
            ], align="center"),
            href="#",
        ),
    ]),
    color="primary",
    dark=True,
)

about_card = dbc.Card([
    dbc.CardHeader("About This Dashboard"),
    dbc.CardBody(
        html.P([
            """This dashboard fetches articles from The Guardian's API, 
            processes them with Natural Language Processing (LDA topic modeling), 
            and visualizes how news topics emerge and evolve over time.""",
            html.A(
                "Learn more on GitHub",
                href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model",
                target="_blank"
            )
        ])
    )
])

date_controls = dbc.Card([
    dbc.CardHeader("Date Range Selection"),
    dbc.CardBody([
        dbc.ButtonGroup([
            dbc.Button("Last Day", id="date-1d", n_clicks=0, color="outline-primary"),
            dbc.Button("Last 3 Days", id="date-3d", n_clicks=0, color="outline-primary"),
            dbc.Button("Last Week", id="date-7d", n_clicks=0, color="primary"),
        ]),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Start Date", html_for="start-date"),
                dbc.Input(
                    type="date", 
                    id="start-date", 
                    value=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                ),
            ]),
            dbc.Col([
                dbc.Label("End Date", html_for="end-date"),
                dbc.Input(
                    type="date", 
                    id="end-date", 
                    value=datetime.now().strftime('%Y-%m-%d')
                ),
            ]),
        ]),
    ])
])

topic_controls = dbc.Card([
    dbc.CardHeader("LDA Topic Settings"),
    dbc.CardBody([
        dbc.Label("Number of Topics", html_for="num-topics-slider"),
        dcc.Slider(
            id="num-topics-slider",
            min=2,
            max=10,
            step=1,
            value=5,
            marks={i: str(i) for i in range(2, 11)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ])
])

tsne_controls = dbc.Card([
    dbc.CardHeader("t-SNE Visualization Settings"),
    dbc.CardBody([
        dbc.Label("Perplexity", html_for="tsne-perplexity-slider"),
        dcc.Slider(
            id="tsne-perplexity-slider",
            min=5,
            max=40,
            step=5,
            value=30,
            marks={i: str(i) for i in range(5, 41, 5)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ])
])

topic_selector = dbc.Card([
    dbc.CardHeader("Topic Explorer"),
    dbc.CardBody([
        html.Div(id="topic-buttons"),
        html.P("Select a topic to focus visualizations.")
    ])
], id="topic-selector-card", style={"display": "none"})

topic_distribution_card = dbc.Card([
    dbc.CardHeader("Topic Word Distributions"),
    dbc.CardBody(
        dcc.Loading(
            dcc.Graph(id="topic-distribution", config={'displayModeBar': True})
        )
    )
])

word_cloud_card = dbc.Card([
    dbc.CardHeader("Topic Word Cloud"),
    dbc.CardBody(
        dcc.Loading(
            dcc.Graph(id="word-cloud", config={'displayModeBar': False})
        )
    )
])

tsne_card = dbc.Card([
    dbc.CardHeader("3D Topic Clustering (t-SNE)"),
    dbc.CardBody(
        dcc.Loading(
            dcc.Graph(id="tsne-plot", config={'displayModeBar': True})
        )
    )
])

bubble_chart_card = dbc.Card([
    dbc.CardHeader("Document Length Over Time"),
    dbc.CardBody(
        dcc.Loading(
            dcc.Graph(id="bubble-chart", config={'displayModeBar': True})
        )
    )
])

ngram_chart_card = dbc.Card([
    dbc.CardHeader("Bigrams & Trigrams (Radar)"),
    dbc.CardBody(
        dcc.Loading(
            dcc.Graph(id="ngram-chart", config={'displayModeBar': True})
        )
    )
])

topic_network_card = dbc.Card([
    dbc.CardHeader("Topic Network"),
    dbc.CardBody(
        dcc.Loading(
            dcc.Graph(id="topic-network", config={'displayModeBar': True})
        )
    )
])

article_table_card = dbc.Card([
    dbc.CardHeader("Article Details"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Div(id="article-count"), width=8),
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText("Filter by topic"),
                    dbc.Select(
                        id="topic-filter",
                        options=[
                            {"label": "All Topics", "value": "all"}
                        ],
                        value="all"
                    )
                ])
            ], width=4)
        ]),
        html.Br(),
        dash_table.DataTable(
            id="article-table",
            columns=[
                {"name": "Title", "id": "title", "presentation": "markdown"},
                {"name": "Published", "id": "published"},
                {"name": "Topics", "id": "topics", "presentation": "markdown"},
            ],
            page_size=10,
            page_action="native",
            sort_action="native",
            filter_action="native",
        ),
    ]),
])

# Simplified app layout
app.layout = html.Div([
    dcc.Store(id="app-data"),
    dcc.Interval(id="auto-load", interval=1000, n_intervals=0),  # Auto-load trigger
    
    navbar,
    dbc.Container([
        html.Br(),
        dbc.Row([
            dbc.Col([
                about_card,
                html.Br(),
                dbc.Row([
                    dbc.Col(date_controls, md=4),
                    dbc.Col(topic_controls, md=4),
                    dbc.Col(tsne_controls, md=4),
                ]),
                html.Br(),
                topic_selector,
                html.Br(),
                
                # All visualizations in one tab for simplicity
                dbc.Row([
                    dbc.Col(topic_distribution_card, md=6),
                    dbc.Col(word_cloud_card, md=6),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(tsne_card, md=6),
                    dbc.Col(bubble_chart_card, md=6),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(ngram_chart_card, md=6),
                    dbc.Col(topic_network_card, md=6),
                ]),
                html.Br(),
                article_table_card,
                
                html.Footer([
                    html.Hr(),
                    html.P([
                        "Data sourced from ",
                        html.A("The Guardian Open Platform", href="https://open-platform.theguardian.com/", target="_blank"),
                        ". This dashboard is for educational purposes only."
                    ], style={"textAlign": "center"})
                ])
            ])
        ])
    ], fluid=True)
])

# Callbacks
@app.callback(
    [Output("start-date", "value"), Output("end-date", "value")],
    [
        Input("date-1d", "n_clicks"),
        Input("date-3d", "n_clicks"),
        Input("date-7d", "n_clicks"),
    ],
)
def update_date_range(n1, n3, n7):
    ctx = callback_context
    if not ctx.triggered:
        # Default to 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    end_date = datetime.now()
    
    if button_id == "date-1d":
        start_date = end_date - timedelta(days=1)
    elif button_id == "date-3d":
        start_date = end_date - timedelta(days=3)
    elif button_id == "date-7d":
        start_date = end_date - timedelta(days=7)
    else:
        # Default
        start_date = end_date - timedelta(days=7)
        
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Auto-load data when inputs change
@app.callback(
    [Output("app-data", "data"),
     Output("topic-buttons", "children"),
     Output("topic-selector-card", "style"),
     Output("topic-filter", "options"),
     Output("article-count", "children")],
    [Input("start-date", "value"),
     Input("end-date", "value"),
     Input("num-topics-slider", "value"),
     Input("tsne-perplexity-slider", "value"),
     Input("auto-load", "n_intervals")]
)
def auto_load_data(start_date, end_date, num_topics, perplexity, n_intervals):
    # Process articles
    df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date, num_topics)
    
    if df is None or df.empty:
        return None, [], {"display": "none"}, [{"label": "All Topics", "value": "all"}], "No articles found for the selected date range."
    
    # Process document lengths and dominant topics
    doc_lengths = []
    doc_dominant_topics = []
    doc_topic_distributions = []
    
    for i in df.index:
        doc_topics = lda_model.get_document_topics(corpus[i])
        n_tokens = len(texts[i] if texts[i] else [])
        doc_lengths.append(n_tokens)
        
        # Store dominant topic
        if doc_topics:
            best_t = max(doc_topics, key=lambda x: x[1])[0]
        else:
            best_t = -1
        doc_dominant_topics.append(best_t)
        
        # Store topic distribution
        topic_dist = {t_id: 0.0 for t_id in range(num_topics)}
        for t_id, weight in doc_topics:
            topic_dist[t_id] = weight
        doc_topic_distributions.append(topic_dist)
    
    df["doc_length"] = doc_lengths
    df["dominant_topic"] = doc_dominant_topics
    
    # Extract topic word distributions
    topic_distributions = {}
    for t_id in range(num_topics):
        topic_distributions[t_id] = lda_model.show_topic(t_id, topn=10)
    
    # Create word clouds for each topic
    word_clouds = {}
    for t_id in range(num_topics):
        word_clouds[t_id] = lda_model.show_topic(t_id, topn=30)
    
    # Extract ngrams
    ngram_counts = {}
    for tokens in texts:
        for tok in tokens:
            if "_" in tok:
                ngram_counts[tok] = ngram_counts.get(tok, 0) + 1
    
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    top_ngrams = sorted_ngrams[:15]
    
    # Prepare ngram data
    formatted_ngrams = []
    for ngram, count in top_ngrams:
        formatted = ngram.replace('_', ' ')
        formatted_ngrams.append({"ngram": formatted, "count": count})
    
    # Prepare article data for table
    articles = []
    for i in df.index:
        doc_topics = lda_model.get_document_topics(corpus[i])
        topics_formatted = []
        for t_id, weight in sorted(doc_topics, key=lambda x: x[1], reverse=True)[:3]:  # Top 3 topics
            topics_formatted.append(f"**Topic {t_id}**: {weight:.3f}")
        
        articles.append({
            "title": df.at[i, "title"],
            "published": df.at[i, "published"].strftime("%Y-%m-%d %H:%M"),
            "topics": "<br>".join(topics_formatted),
            "doc_length": df.at[i, "doc_length"],
            "dominant_topic": int(df.at[i, "dominant_topic"]),
            "topic_distribution": doc_topic_distributions[i]
        })
    
    # Prepare t-SNE data
    doc_topics_list = []
    for i in df.index:
        topic_weights = [0.0] * lda_model.num_topics
        for topic_id, w in lda_model.get_document_topics(corpus[i]):
            topic_weights[topic_id] = w
        doc_topics_list.append(topic_weights)
    
    doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
    
    perplex_val = min(perplexity, max(2, len(doc_topics_array) - 1))
    tsne = TSNE(
        n_components=3,  # Keeping 3D t-SNE
        random_state=42,
        perplexity=perplex_val,
        n_jobs=1,
        n_iter=250,
        learning_rate='auto'
    )
    embedded = tsne.fit_transform(doc_topics_array)
    
    tsne_data = {
        "x": embedded[:, 0].tolist(),
        "y": embedded[:, 1].tolist(),
        "z": embedded[:, 2].tolist(),  # Keeping z dimension for 3D
        "topic": [int(t) for t in doc_dominant_topics],
        "title": df["title"].tolist()
    }
    
    # Store all data
    data = {
        "articles": articles,
        "topic_distributions": topic_distributions,
        "word_clouds": word_clouds,
        "tsne_data": tsne_data,
        "ngrams": formatted_ngrams,
        "num_topics": num_topics,
        "start_date": start_date,
        "end_date": end_date,
        "perplexity": perplexity
    }
    
    # Create topic buttons
    buttons = [
        html.Button(
            "All Topics",
            id="topic-btn-all",
            style={"margin": "5px", "backgroundColor": "#005689", "color": "white"},
            n_clicks=0
        )
    ]
    
    for i in range(num_topics):
        buttons.append(
            html.Button(
                f"Topic {i}",
                id=f"topic-btn-{i}",
                style={"margin": "5px"},
                n_clicks=0
            )
        )
    
    # Create topic filter options
    filter_options = [{"label": "All Topics", "value": "all"}]
    for i in range(num_topics):
        filter_options.append({"label": f"Topic {i}", "value": str(i)})
    
    # Show topic selector
    selector_style = {"display": "block"}
    
    # Article count message
    count_message = html.Div([
        html.Strong(f"Found {len(articles)} articles"),
        html.Span(f" from {start_date} to {end_date}")
    ])
    
    return data, buttons, selector_style, filter_options, count_message

# Update visualizations based on app data
@app.callback(
    [Output("topic-distribution", "figure"),
     Output("word-cloud", "figure"),
     Output("tsne-plot", "figure"),
     Output("bubble-chart", "figure"),
     Output("ngram-chart", "figure"),
     Output("topic-network", "figure"),
     Output("article-table", "data"),
     Output("topic-filter", "value")],
    [Input("app-data", "data"),
     Input("topic-btn-all", "n_clicks")] +
    [Input(f"topic-btn-{i}", "n_clicks") for i in range(10)] +  # Support up to 10 topics
    [Input("topic-filter", "value")]
)
def update_visualizations(data, *args):
    # Extract n_clicks and filter
    n_clicks_list = args[:11]  # First 11 args are topic button n_clicks
    filter_value = args[11]    # Last arg is topic-filter value
    
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Determine selected topic
    selected_topic = "all"
    if triggered_id and triggered_id.startswith("topic-btn-"):
        selected_topic = triggered_id.split('-')[-1]
    elif triggered_id == "topic-filter":
        selected_topic = filter_value
    
    # Default empty figures
    empty_fig = go.Figure()
    empty_fig.update_layout(title="No data available")
    
    if data is None:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], "all"
    
    # Update topic filter if a topic button was clicked
    new_filter_value = filter_value
    if triggered_id and triggered_id.startswith("topic-btn-"):
        new_filter_value = selected_topic
    
    # Topic distributions visualization
    try:
        topic_dist_data = []
        for topic_id, words in data["topic_distributions"].items():
            for word, weight in words:
                topic_dist_data.append({
                    "topic": f"Topic {topic_id}",
                    "word": word,
                    "weight": weight
                })
        
        df_topic_dist = pd.DataFrame(topic_dist_data)
        
        topic_dist_fig = px.bar(
            df_topic_dist,
            x="weight",
            y="word",
            color="topic",
            orientation="h",
            title="Topic Word Distributions (Top Terms)",
            labels={"weight": "Term Weight", "word": "", "topic": "Topic"},
            height=500
        )
        
        topic_dist_fig.update_layout(**get_plot_layout(""))
        topic_dist_fig.update_layout(yaxis=dict(autorange="reversed"))
    except Exception as e:
        logger.error(f"Error creating topic distribution: {e}", exc_info=True)
        topic_dist_fig = empty_fig
    
    # Word cloud visualization
    try:
        if selected_topic == "all":
            # For "all", use topic 0
            word_cloud_topic = 0
        else:
            word_cloud_topic = int(selected_topic)
        
        if word_cloud_topic < len(data["word_clouds"]):
            word_cloud_fig = create_word_cloud(data["word_clouds"][str(word_cloud_topic)])
        else:
            word_cloud_fig = empty_fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        word_cloud_fig = empty_fig
    
    # 3D t-SNE visualization
    try:
        tsne_fig = go.Figure(data=[
            go.Scatter3d(
                x=data["tsne_data"]["x"],
                y=data["tsne_data"]["y"],
                z=data["tsne_data"]["z"],  # Using z dimension for 3D
                mode="markers",
                marker=dict(
                    size=5,
                    color=data["tsne_data"]["topic"],
                    colorscale="Viridis",
                    opacity=0.8
                ),
                text=data["tsne_data"]["title"],
                hoverinfo="text"
            )
        ])
        
        tsne_fig.update_layout(**get_plot_layout("3D Topic Clustering (t-SNE)"))
        tsne_fig.update_layout(
            scene=dict(
                xaxis=dict(title="", showticklabels=False),
                yaxis=dict(title="", showticklabels=False),
                zaxis=dict(title="", showticklabels=False)
            )
        )
    except Exception as e:
        logger.error(f"Error creating t-SNE: {e}", exc_info=True)
        tsne_fig = empty_fig
    
    # Bubble chart visualization
    try:
        bubble_data = data["articles"]
        if selected_topic != "all":
            bubble_data = [a for a in bubble_data if str(a["dominant_topic"]) == selected_topic]
        
        bubble_df = pd.DataFrame(bubble_data)
        if not bubble_df.empty:
            bubble_df["published"] = pd.to_datetime(bubble_df["published"])
            
            bubble_fig = px.scatter(
                bubble_df,
                x="published",
                y="doc_length",
                size="doc_length",
                color="dominant_topic",
                size_max=20,
                hover_name="title",
                title="Document Length Over Time",
                labels={
                    "published": "Publication Date",
                    "doc_length": "Article Length (tokens)",
                    "dominant_topic": "Dominant Topic"
                }
            )
            
            bubble_fig.update_layout(**get_plot_layout(""))
            bubble_fig.update_layout(
                yaxis=dict(type="log"),
                xaxis=dict(
                    title="Publication Date",
                    tickformat="%d %b %Y"
                )
            )
        else:
            bubble_fig = empty_fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        bubble_fig = empty_fig
    
    # Ngram radar chart
    try:
        ngram_df = pd.DataFrame(data["ngrams"])
        
        ngram_fig = px.line_polar(
            ngram_df,
            r="count",
            theta="ngram",
            line_close=True,
            title="Top Bigrams & Trigrams"
        )
        
        ngram_fig.update_traces(
            fill='toself',
            fillcolor=f"rgba({0},{86},{137},{0.3})"  # Semi-transparent guardian blue
        )
        
        ngram_fig.update_layout(**get_plot_layout(""))
        ngram_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(ngram_df["count"]) * 1.1]
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=False
        )
    except Exception as e:
        logger.error(f"Error creating ngram chart: {e}", exc_info=True)
        ngram_fig = empty_fig
    
    # Topic network chart
    try:
        if "topic_similarities" in data:
            topic_network_fig = create_topic_network(
                data["topic_similarities"], 
                {"num_topics": data["num_topics"], 
                 "show_topic": lambda i, n: data["topic_distributions"][str(i)][:n]}
            )
        else:
            topic_network_fig = empty_fig
    except Exception as e:
        logger.error(f"Error creating topic network: {e}", exc_info=True)
        topic_network_fig = empty_fig
    
    # Article table data
    try:
        articles = data["articles"]
        if selected_topic != "all":
            # Filter by selected topic
            articles = [a for a in articles if str(a["dominant_topic"]) == selected_topic]
            
        # Use markdown formatting for titles and topics
        for a in articles:
            # Wrap title in a clickable link that opens in a new tab
            a["title"] = f"[{a['title']}](https://www.theguardian.com/search?q={a['title'].replace(' ', '+')}) <i class='fas fa-external-link-alt' style='font-size: 0.8em'></i>"
    except Exception as e:
        logger.error(f"Error preparing article table: {e}", exc_info=True)
        articles = []
    
    return (
        topic_dist_fig,
        word_cloud_fig,
        tsne_fig,
        bubble_fig,
        ngram_fig,
        topic_network_fig,
        articles,
        new_filter_value
    )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=False)
