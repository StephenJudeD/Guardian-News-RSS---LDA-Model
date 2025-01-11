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
# GuardianFetcher (expects a .py that fetches articles)
# ─────────────────────────────────────────────────────────────────────
guardian = GuardianFetcher(GUARDIAN_API_KEY)

# ─────────────────────────────────────────────────────────────────────
# Dash App Setup
# ─────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])
server = app.server
app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def process_articles(start_date, end_date):
    """
    Fetch articles from Guardian, filter by date,
    tokenize, train LDA, return (df, texts, dictionary, corpus, lda_model).
    We keep LDA passes low to avoid hitting Heroku 30s timeouts.
    """
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date}")
        
        # Date parsing
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        days_back = (datetime.now().date() - start_date_dt).days + 1
        
        # Fetch the articles
        df = guardian.fetch_articles(days_back=days_back, page_size=200)
        if df.empty:
            logger.warning("No articles fetched!")
            return None, None, None, None, None
        
        # Filter to date range
        df = df[
            (df['published'].dt.date >= start_date_dt) &
            (df['published'].dt.date <= end_date_dt)
        ]
        logger.info(f"Filtered to {len(df)} articles within date range")
        if len(df) < 5:
            logger.warning("Not enough articles for meaningful LDA.")
            return None, None, None, None, None
        
        # Reset index so df.index is 0..(len(df)-1)
        df.reset_index(drop=True, inplace=True)
        
        # Tokenize and remove stop words
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
        
        # Gensim dictionary + corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(t) for t in texts]
        
        # Train LDA with fewer passes => faster on Heroku
        # e.g. passes=5 or 10 instead of 20
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
        logger.error(f"Error in process_articles: {e}", exc_info=True)
        return None, None, None, None, None

# ─────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────
def create_word_cloud(topic_words):
    """
    Build a word cloud from topic words (list of (word, prob) pairs).
    """
    try:
        freq_dict = dict(topic_words)
        wc = WordCloud(
            background_color='black',
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

def create_tsne_visualization(df, corpus, lda_model):
    """
    Construct a t-SNE scatter plot from LDA topic vectors.
    Use n_jobs=1 to avoid CPU detection errors on Heroku.
    """
    try:
        if df is None or len(df) < 2:
            return go.Figure().update_layout(
                template='plotly',
                title='Not enough documents for t-SNE'
            )
        
        # Build doc_topics from corpus
        doc_topics_list = []
        for i in df.index:  # i = 0..(len(df)-1)
            topic_weights = [0.0]*lda_model.num_topics
            for topic_id, w in lda_model[corpus[i]]:
                topic_weights[topic_id] = w
            doc_topics_list.append(topic_weights)
        
        doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
        # If fewer than 2 docs => skip
        if len(doc_topics_array) < 2:
            return go.Figure().update_layout(
                template='plotly',
                title='Not enough docs for t-SNE'
            )
        
        # Adjust perplexity
        perplex_val = 30
        if len(doc_topics_array) < 30:
            perplex_val = max(2, len(doc_topics_array) - 1)
        
        # Set n_jobs=1 to avoid CPU detection errors
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplex_val,
            n_jobs=1
        )
        embedded = tsne.fit_transform(doc_topics_array)
        
        scatter_df = pd.DataFrame({
            'x': embedded[:, 0],
            'y': embedded[:, 1],
            'dominant_topic': [np.argmax(arr) + 1 for arr in doc_topics_array],
            'doc_index': df.index,
            'title': df['title']
        })
        
        fig = px.scatter(
            scatter_df,
            x='x', y='y',
            color='dominant_topic',
            hover_data=['title'],
            title='t-SNE Topic Clustering',
            custom_data=['doc_index']
        )
        fig.update_layout(template='plotly')
        return fig
    except Exception as e:
        logger.error(f"Error creating t-SNE: {e}", exc_info=True)
        return go.Figure().update_layout(template='plotly', title=f"t-SNE Error: {e}")

# ─────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────
navbar = dbc.NavbarSimple(
    brand="Guardian News Topic Explorer",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

banner = dbc.Container(
    [
        html.H1("Guardian News Topic Explorer 📰", className="display-3 fw-bold"),
        html.P(
            "Interactive topic modeling and clustering of Guardian articles. Fewer LDA passes for faster load on Heroku.",
            className="lead text-muted"
        ),
    ],
    fluid=True,
    className="py-5 my-4 bg-light rounded-3 text-center"
)

controls_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Controls")),
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
                className="mb-3"
            ),
            dcc.Dropdown(
                id='topic-filter',
                options=[{'label': f'Topic {i+1}', 'value': i} for i in range(5)],
                multi=True,
                placeholder="Filter by topics...",
                className="mb-3"
            )
        ])
    ],
    className="mb-4 shadow"
)

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
            dcc.Graph(id='word-cloud', style={"height": "400px"})
        ])
    ],
    className="mb-4 shadow"
)

tsne_card = dbc.Card(
    [
        dbc.CardHeader("t-SNE Topic Clustering", className="bg-secondary text-light"),
        dbc.CardBody([
            dcc.Graph(id='tsne-plot', style={"height": "600px"})
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
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': 'rgb(50,50,50)',
                    'color': 'white',
                    'textAlign': 'left',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_header={
                    'backgroundColor': 'rgb(30,30,30)',
                    'fontWeight': 'bold'
                },
                page_size=10
            )
        ])
    ],
    className="mb-4 shadow"
)

app.layout = dbc.Container([
    navbar,
    banner,
    dbc.Row([
        dbc.Col(controls_card, md=4),
        dbc.Col([
            topic_distribution_card,
            word_cloud_card
        ], md=8)
    ], align="start"),
    dbc.Row([dbc.Col(tsne_card, md=12)]),
    dbc.Row([dbc.Col(articles_table_card, md=12)])
], fluid=True)

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
    Sync date picker with radio items for convenience.
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
    1) Pull data from process_articles
    2) Build topic distribution, word cloud for the 1st selected topic
    3) t-SNE scatter
    4) Article table with selected topics
    """
    try:
        logger.info(f"Updating from {start_date} to {end_date}, topics={selected_topics}")
        df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date)
        if df is None or df.empty:
            # return placeholders
            logger.error("No data returned from process_articles.")
            empty_fig = go.Figure().update_layout(template='plotly')
            return empty_fig, empty_fig, empty_fig, []
        
        if not selected_topics:
            selected_topics = list(range(lda_model.num_topics))
        
        # Create bar chart of top words for each selected topic
        terms = []
        for t_id in selected_topics:
            topn = lda_model.show_topic(t_id, topn=10)
            for word, prob in topn:
                terms.append((word, prob, t_id))
        
        if not terms:
            # fallback if user picks invalid topics
            dist_fig = go.Figure().update_layout(template='plotly', title='No topics selected')
        else:
            tmp_df = pd.DataFrame(terms, columns=['word', 'probability', 'topic'])
            dist_fig = px.bar(
                tmp_df,
                x='probability',
                y='word',
                color='topic',
                orientation='h',
                title='Topic Word Distributions'
            )
            dist_fig.update_layout(template='plotly')
        
        # Word cloud from the first selected topic
        first_t = selected_topics[0]
        wcloud_fig = create_word_cloud(lda_model.show_topic(first_t, topn=30))
        
        # t-SNE scatter
        tsne_fig = create_tsne_visualization(df, corpus, lda_model)
        
        # Build the table data
        # Attach topic distribution for each doc, filtering to selected topics
        table_rows = []
        for i in df.index:
            doc_topics = lda_model.get_document_topics(corpus[i])
            row_topics = []
            for t_id, weight in sorted(doc_topics, key=lambda x: x[1], reverse=True):
                if t_id in selected_topics:
                    row_topics.append(f"Topic {t_id+1}: {weight:.3f}")
            table_rows.append({
                'title': df.at[i, 'title'],
                'section': df.at[i, 'section'],
                'published': df.at[i, 'published'].strftime('%Y-%m-%d %H:%M'),
                'topics': '\n'.join(row_topics)
            })
        
        return dist_fig, wcloud_fig, tsne_fig, table_rows
    
    except Exception as e:
        logger.error(f"Main callback error: {e}", exc_info=True)
        empty_fig = go.Figure().update_layout(template='plotly')
        return empty_fig, empty_fig, empty_fig, []

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
