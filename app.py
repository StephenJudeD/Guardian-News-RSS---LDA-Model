import dash
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from gensim.models import CoherenceModel
from wordcloud import WordCloud
from sklearn.manifold import TSNE
import numpy as np
import requests  # Import the requests library

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
# Environment variables
# ─────────────────────────────────────────────────────────────────────
load_dotenv()
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
if not GUARDIAN_API_KEY:
    logger.error("⚠️ No GUARDIAN_API_KEY found in environment!")
    # Display a user-friendly error message in the app
    api_key_error_message = "⚠️ No GUARDIAN_API_KEY found in environment!  Please set the GUARDIAN_API_KEY environment variable."
else:
    api_key_error_message = None

# ─────────────────────────────────────────────────────────────────────
# NLTK Downloads
# ─────────────────────────────────────────────────────────────────────
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

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

stop_words = set(stopwords.words('english')).union(CUSTOM_STOP_WORDS)

# ─────────────────────────────────────────────────────────────────────
# GuardianFetcher (Using API Key)
# ─────────────────────────────────────────────────────────────────────
class GuardianFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com/search"

    def fetch_articles(self, days_back=7, page_size=200):
        """Fetches articles from the Guardian API."""
        all_articles = []
        current_date = datetime.now().date()
        end_date = current_date
        begin_date = current_date - timedelta(days=days_back)

        page = 1
        total_pages = 1  # Initialize to 1 to enter the loop

        while page <= total_pages:
            params = {
                'api-key': self.api_key,
                'from-date': begin_date.strftime('%Y-%m-%d'),
                'to-date': end_date.strftime('%Y-%m-%d'),
                'page': page,
                'page-size': page_size,
                'show-fields': 'headline,bodyText,firstPublicationDate'
            }

            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                if 'response' in data and 'results' in data['response']:
                    results = data['response']['results']
                    total_pages = data['response']['pages']  # Update total_pages
                    for item in results:
                        article = {
                            'title': item['fields']['headline'] if 'headline' in item['fields'] else 'No Headline',
                            'content': item['fields']['bodyText'] if 'bodyText' in item['fields'] else 'No Content',
                            'published': pd.to_datetime(item['fields']['firstPublicationDate']) if 'firstPublicationDate' in item['fields'] else None
                        }
                        all_articles.append(article)
                    logger.info(f"Fetched page {page}/{total_pages} from Guardian API")
                    page += 1  # Increment page number
                else:
                    logger.warning("No 'response' or 'results' found in API response.")
                    break  # Exit loop if no results

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                break  # Exit loop on request error
            except Exception as e:
                logger.error(f"Error processing API response: {e}")
                break  # Exit loop on processing error

        df = pd.DataFrame(all_articles)
        return df

# ─────────────────────────────────────────────────────────────────────
# Data Processing Function
# ─────────────────────────────────────────────────────────────────────
def process_articles(start_date, end_date, num_topics=5, lda_passes=10):
    """
    Fetches, preprocesses, and trains an LDA model on Guardian articles.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        num_topics (int): Number of topics for LDA.
        lda_passes (int): Number of passes for LDA training.

    Returns:
        tuple: (df, texts, dictionary, corpus, lda_model, coherence_score) or (None,)*6 on error.
    """
    try:
        logger.info(f"Fetching articles from {start_date} to {end_date}")

        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        days_back = (datetime.now().date() - start_date_dt).days + 1

        guardian = GuardianFetcher(GUARDIAN_API_KEY)  # Initialize here
        df = guardian.fetch_articles(days_back=days_back, page_size=200)

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
            num_topics=num_topics,
            id2word=dictionary,
            passes=lda_passes,
            random_state=42,
            chunksize=100
        )

        # Calculate Coherence
        coherence_score = calculate_coherence(lda_model, texts, dictionary)

        logger.info(f"Processed {len(df)} articles successfully")
        return df, texts, dictionary, corpus, lda_model, coherence_score

    except Exception as e:
        logger.error(f"Error in process_articles: {e}", exc_info=True)
        return None, None, None, None, None, None

# ─────────────────────────────────────────────────────────────────────
# Coherence Calculation Function
# ─────────────────────────────────────────────────────────────────────
def calculate_coherence(lda_model, texts, dictionary):
    """
    Calculates the coherence score of an LDA model.

    Args:
        lda_model: Trained LDA model.
        texts: List of tokenized texts.
        dictionary: Gensim dictionary.

    Returns:
        float: Coherence score.
    """
    try:
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        logger.info(f"Coherence score: {coherence_score}")
        return coherence_score
    except Exception as e:
        logger.error(f"Error calculating coherence: {e}", exc_info=True)
        return None

# ─────────────────────────────────────────────────────────────────────
# Visualization Functions
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
    3D t-SNE scatter (Plotly).
    Uses all documents in df/corpus to avoid filtering by topic.
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
            hover_data=['title']
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
        return go.Figure().update_layout(template='plotly', title=str(e))

# ─────────────────────────────────────────────────────────────────────
# Visualization Helper Functions (Moved Inside app.py for Simplicity)
# ─────────────────────────────────────────────────────────────────────
def create_topic_distribution_chart(lda_model, selected_topics):
    """
    Creates a bar chart of topic word distributions.
    """
    words_list = []
    for t_id in selected_topics:
        top_pairs = lda_model.show_topic(t_id, topn=20)
        for (w, prob) in top_pairs:
            words_list.append((w, prob, t_id))

    if not words_list:
        return go.Figure().update_layout(template='plotly', title="No topics found")

    df_dist = pd.DataFrame(words_list, columns=["word", "prob", "topic"])
    fig = px.bar(
        df_dist,
        x="prob",
        y="word",
        color="topic",
        orientation="h",
        title="Topic Word Distributions"
    )
    fig.update_layout(template='plotly', yaxis={'categoryorder': 'total ascending'})
    return fig

# ─────────────────────────────────────────────────────────────────────
# Theming (Guardian-like)
# ─────────────────────────────────────────────────────────────────────
NAVY_BLUE = "#052962"

# ─────────────────────────────────────────────────────────────────────
# Dash Setup
# ─────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    html.H1("Guardian News Topic Explorer"),

    # API Key Error Message
    html.Div(api_key_error_message, style={'color': 'red'}) if api_key_error_message else "",

    # Date Range Selection
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=(datetime.now() - timedelta(days=14)).date(),
            end_date=datetime.now().date()
        )
    ]),

    # Number of Topics Input
    html.Div([
        html.Label("Number of Topics:"),
        dcc.Input(
            id='num-topics-input',
            type='number',
            value=5,
            min=2,
            max=10,
            step=1
        )
    ]),

    # LDA Passes Input
    html.Div([
        html.Label("LDA Passes:"),
        dcc.Input(
            id='lda-passes-input',
            type='number',
            value=10,
            min=5,
            max=50,
            step=5
        )
    ]),

    # Topic Filter
    html.Div([
        html.Label("Select Topics:"),
        dcc.Dropdown(
            id='topic-filter',
            options=[{'label': f'Topic {i}', 'value': i} for i in range(10)],  # Adjust range
            multi=True,
            placeholder="Filter by topics..."
        )
    ]),

    # Coherence Score Display
    html.Div(id='coherence-score'),

    # Topic Word Distribution
    dcc.Graph(id='topic-distribution'),

    # Word Cloud
    dcc.Graph(id='word-cloud'),

    # 3D t-SNE Plot
    dcc.Graph(id='tsne-plot'),

    # Bubble Chart
    dcc.Graph(id='bubble-chart'),

    # Bigrams and Trigrams Chart
    dcc.Graph(id='bigrams-trigrams'),

    # Article Table
    dash_table.DataTable(
        id='article-details',
        columns=[
            {'name': 'Title', 'id': 'title'},
            {'name': 'Published', 'id': 'published'},
            {'name': 'Topics', 'id': 'topics'}
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
], fluid=True)

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
        Output('article-details', 'data'),
        Output('coherence-score', 'children')
    ],
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('num-topics-input', 'value'),
        Input('lda-passes-input', 'value'),
        Input('topic-filter', 'value')
    ]
)
def update_visuals(start_date, end_date, num_topics, lda_passes, selected_topics):
    """
    Main callback to update visuals based on user inputs.
    """
    try:
        logger.info(f"Updating visuals: {start_date}, {end_date}, {num_topics}, {lda_passes}, {selected_topics}")

        if not GUARDIAN_API_KEY:
            return [go.Figure().update_layout(title="API Key Missing")] * 6 + ["API Key Missing"]

        df, texts, dictionary, corpus, lda_model, coherence_score = process_articles(
            start_date, end_date, num_topics, lda_passes
        )

        if df is None or df.empty:
            return [go.Figure().update_layout(title="No Data")] * 6 + ["No Data"]

        # Default topics if none selected
        if not selected_topics:
            selected_topics = list(range(num_topics))

        # Build doc-level info
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

        # Filter DataFrame
        filtered_df = df[df["dominant_topic"].isin(selected_topics)].copy()
        if filtered_df.empty:
            return [go.Figure().update_layout(title="No Data (Topic Filtered)")] * 6 + ["No Data (Topic Filtered)"]

        filtered_texts = [texts[i] for i in filtered_df.index]
        filtered_corpus = [corpus[i] for i in filtered_df.index]

        # Create Visualizations
        topic_dist_fig = create_topic_distribution_chart(lda_model, selected_topics)
        wc_fig = create_word_cloud(lda_model.show_topic(selected_topics[0], topn=30)) if selected_topics else go.Figure() # Word cloud for first selected topic
        tsne_fig = create_tsne_visualization_3d(df, corpus, lda_model)
        bubble_fig = create_bubble_chart(filtered_df)
        ngram_fig = create_ngram_bar_chart(filtered_texts)

        # Article Table Data
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

        return topic_dist_fig, wc_fig, tsne_fig, bubble_fig, ngram_fig, table_data, f"Coherence Score: {coherence_score:.4f}"

    except Exception as e:
        logger.error(f"Error in update_visuals: {e}", exc_info=True)
        error_message = f"Error: {e}"
        return [go.Figure().update_layout(title=error_message)] * 6 + [error_message]

# ─────────────────────────────────────────────────────────────────────
# Run the App
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8050))
    app.run_server(debug=True, port=port)
