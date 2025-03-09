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
from gensim.models.coherencemodel import CoherenceModel

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import nltk
import os
from dotenv import load_dotenv
from functools import lru_cache
import logging
import textwrap
import networkx as nx

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
    'breaking', 'update', 'live', 'say', 'going', 'think', 'know', 'just', 'now', 'even', 'taking', 'back'
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
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
app = Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server
app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────
# Guardian Theme & Dark Mode
# ─────────────────────────────────────────────────────────────────────
GUARDIAN_COLORS = {
    "blue": "#005689",
    "blue_light": "#00b2ff",
    "red": "#c70000",
    "yellow": "#ffbb00",
    "background": "#f6f6f6",
    "border": "#dcdcdc",
}

def get_guardian_plot_layout(fig_title="", dark_mode=False):
    """Return a default layout for Guardian-themed figures, with dark mode support."""
    return dict(
        paper_bgcolor="#1e1e1e" if dark_mode else "white",
        plot_bgcolor="#2d2d2d" if dark_mode else "#f6f6f6",
        font=dict(
            family="Georgia, serif", 
            size=16,
            color="white" if dark_mode else "#333333"
        ),
        title=dict(text=fig_title, font=dict(size=20)),
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(
            family="Georgia, serif", 
            size=20, 
            color="#00b2ff" if dark_mode else "#005689"
        ),
        legend_title_font=dict(family="Georgia, serif", size=16),
        legend_font=dict(family="Georgia, serif", size=14),
        colorway=[
            "#005689", "#c70000", "#ffbb00", "#00b2ff", "#90dcff", 
            "#ff5b5b", "#4bc6df", "#aad801", "#43853d", "#767676"
        ],
        xaxis=dict(
            gridcolor="#444444" if dark_mode else "#dcdcdc",
            zerolinecolor="#444444" if dark_mode else "#dcdcdc",
            showgrid=True,
            showline=True,
            linecolor="#444444" if dark_mode else "#dcdcdc",
            title_font=dict(size=16),
            tickfont=dict(size=14, color="white" if dark_mode else "#333333")
        ),
        yaxis=dict(
            gridcolor="#444444" if dark_mode else "#dcdcdc",
            zerolinecolor="#444444" if dark_mode else "#dcdcdc",
            showgrid=True,
            showline=True,
            linecolor="#444444" if dark_mode else "#dcdcdc",
            title_font=dict(size=16),
            tickfont=dict(size=14, color="white" if dark_mode else "#333333")
        ),
    )

# Dark mode detection script
dark_mode_script = """
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.body.classList.add('dark-mode');
    }
    
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
        if (event.matches) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    });
"""

# Dark mode stylesheet
dark_mode_style = """
    @media (prefers-color-scheme: dark) {
        body.dark-mode {
            background-color: #121212;
            color: #f0f0f0;
        }
        body.dark-mode .card {
            background-color: #1e1e1e;
            border-color: #333;
        }
        body.dark-mode .card-header {
            background-color: #2c2c2c;
            border-color: #333;
        }
        body.dark-mode .table {
            color: #f0f0f0;
        }
        body.dark-mode .table-striped tbody tr:nth-of-type(odd) {
            background-color: rgba(255,255,255,0.05);
        }
        body.dark-mode .nav-link.active {
            background-color: #005689 !important;
            color: white !important;
        }
        body.dark-mode hr {
            border-color: #333;
        }
        body.dark-mode .text-muted {
            color: #aaa !important;
        }
        body.dark-mode .input-group-text {
            background-color: #2c2c2c;
            color: #f0f0f0;
            border-color: #444;
        }
        body.dark-mode .form-control {
            background-color: #2c2c2c;
            color: #f0f0f0;
            border-color: #444;
        }
        body.dark-mode .form-control:focus {
            background-color: #333;
            color: white;
        }
        body.dark-mode .collapse-button {
            background-color: #2c2c2c;
            color: #f0f0f0;
        }
        body.dark-mode .topic-pill {
            background-color: #333;
        }
    }

    /* Tooltip styles */
    .guardian-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .guardian-tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 10;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.875rem;
    }
    
    .guardian-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Topic pills */
    .topic-pill {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        color: white;
    }
    
    /* Collapsible card */
    .collapsible-card .card-header {
        cursor: pointer;
    }
    
    .collapsible-card .collapse-icon {
        transition: transform 0.3s;
    }
    
    .collapsible-card .collapse-icon.expanded {
        transform: rotate(180deg);
    }
    
    /* Card hover effects */
    .hover-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .hover-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Topic buttons */
    .topic-btn {
        margin: 4px;
        transition: all 0.2s;
    }
    
    .topic-btn:hover {
        transform: translateY(-2px);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    .dark-mode ::-webkit-scrollbar-track {
        background: #2d2d2d;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
"""

# ─────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def process_articles(start_date, end_date, num_topics=3):
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
        # Filter extreme values to improve model quality
        dictionary.filter_extremes(no_below=3, no_above=0.85)
        corpus = [dictionary.doc2bow(t) for t in texts]

        # Train LDA with dynamic num_topics
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=5,  # 
            alpha='auto',  # Auto-optimize alpha
            random_state=42,
            chunksize=100,
            per_word_topics=True  # For coherence calculation
        )

        logger.info(f"Processed {len(df)} articles successfully with LDA num_topics={num_topics}")
        return df, texts, dictionary, corpus, lda_model

    except Exception as e:
        logger.error(f"Error in process_articles: {e}", exc_info=True)
        return None, None, None, None, None

# Calculate coherence score for topics
def calculate_coherence(texts, dictionary, lda_model):
    try:
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_per_topic = coherence_model.get_coherence_per_topic()
        
        results = []
        for topic_id, coherence in enumerate(coherence_per_topic):
            results.append({
                'topic': topic_id,
                'coherence': coherence
            })
        return results
    except Exception as e:
        logger.error(f"Error calculating coherence: {e}", exc_info=True)
        return [{'topic': i, 'coherence': 0} for i in range(lda_model.num_topics)]

# Calculate topic similarity matrix
def calculate_topic_similarity(lda_model):
    try:
        num_topics = lda_model.num_topics
        topic_term_dists = np.array([
            [lda_model.expElogbeta[topic, term] for term in range(lda_model.expElogbeta.shape[1])]
            for topic in range(num_topics)
        ])
        
        # Normalize
        topic_term_dists = topic_term_dists / topic_term_dists.sum(axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(topic_term_dists)
        
        # Convert to dictionary format
        result = {}
        for i in range(num_topics):
            result[i] = {}
            for j in range(num_topics):
                if i != j:  # Skip self-similarity
                    result[i][j] = float(similarity_matrix[i, j])
        
        return result
    except Exception as e:
        logger.error(f"Error calculating topic similarity: {e}", exc_info=True)
        return {}

# ─────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────
def create_word_cloud(topic_words, dark_mode=False):
    """
    Create a word cloud from LDA topic-word pairs using Guardian colors.
    """
    try:
        freq_dict = dict(topic_words)
        
        wc = WordCloud(
            background_color="#1e1e1e" if dark_mode else "white",
            width=800,
            height=400,
            colormap="Blues",  # Use blues to match Guardian color scheme
            max_words=50,
            prefer_horizontal=0.9,
            contour_width=1,
            contour_color='#444' if dark_mode else '#ddd'
        ).generate_from_frequencies(freq_dict)

        fig = px.imshow(wc)
        fig.update_layout(**get_guardian_plot_layout("Topic Word Cloud", dark_mode))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating word cloud: {e}", dark_mode))
        return fig

def create_tsne_visualization_3d(df, corpus, lda_model, perplexity=30, dark_mode=False):
    """
    3D t-SNE scatter (Plotly).
    Uses all documents in df/corpus to avoid filtering by topic.
    Allows dynamic perplexity setting.
    """
    try:
        if df is None or len(df) < 2:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("Not enough documents for t-SNE", dark_mode))
            return fig

        doc_topics_list = []
        for i in df.index:
            topic_weights = [0.0] * lda_model.num_topics
            doc_topics, _, _ = lda_model[corpus[i]]  # Unpack all three values
            for topic_id, w in doc_topics:  # Now iterate over the topic distribution
                topic_weights[topic_id] = w
            doc_topics_list.append(topic_weights)

        doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
        if len(doc_topics_array) < 2:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("Not enough docs for t-SNE", dark_mode))
            return fig

        perplex_val = min(perplexity, max(2, len(doc_topics_array) - 1))
        tsne = TSNE(
            n_components=3,
            random_state=42,
            perplexity=perplex_val,
            n_jobs=-1,  # Use all available cores
            learning_rate='auto'  # Improved parameter
        )
        embedded = tsne.fit_transform(doc_topics_array)

        scatter_df = pd.DataFrame({
            'x': embedded[:, 0],
            'y': embedded[:, 1],
            'z': embedded[:, 2],
            'dominant_topic': [np.argmax(row) for row in doc_topics_array],
            'doc_index': df.index,
            'title': df['title'],
            'published': df['published']
        })
        
        # Wrap long titles for better hover display
        scatter_df['hover_title'] = scatter_df['title'].apply(
            lambda x: '<br>'.join(textwrap.wrap(x, width=40))
        )
        scatter_df['hover_text'] = (
            scatter_df['hover_title'] + '<br>' + 
            scatter_df['published'].dt.strftime('%Y-%m-%d %H:%M') + 
            '<br>Topic: ' + scatter_df['dominant_topic'].astype(str)
        )

        fig = px.scatter_3d(
            scatter_df,
            x='x', y='y', z='z',
            color='dominant_topic',
            hover_data=['hover_text'],
            title=f'3D t-SNE Topic Clustering (Perplexity={perplex_val})',
            labels={'dominant_topic': 'Topic'}
        )
        
        # Enhance the plot appearance
        fig.update_traces(
            marker=dict(
                size=6,
                opacity=0.8,
                line=dict(width=0.5, color='white' if dark_mode else 'black')
            ),
            hoverinfo="text",
            hovertemplate="%{customdata[0]}"
        )
        
        fig.update_layout(**get_guardian_plot_layout('', dark_mode))
        # Add camera controls explanation
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='', showticklabels=False),
                yaxis=dict(title='', showticklabels=False),
                zaxis=dict(title='', showticklabels=False),
                annotations=[
                    dict(
                        showarrow=False,
                        x=0.05,
                        y=0.05,
                        z=0.05,
                        text="Drag to rotate, scroll to zoom",
                        font=dict(size=12, color="white" if dark_mode else "black")
                    )
                ]
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating 3D t-SNE: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating 3D t-SNE: {e}", dark_mode))
        return fig

def create_bubble_chart(df, dark_mode=False):
    """
    Bubble chart: doc length vs published date, sized by doc length,
    colored by dominant_topic. Removes outliers & uses log scale.
    """
    try:
        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("Bubble Chart Unavailable", dark_mode))
            return fig

        cut_off = df['doc_length'].quantile(0.95)
        filtered_df = df[df['doc_length'] <= cut_off].copy()
        if filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("No Data after outlier removal", dark_mode))
            return fig

        # Wrap titles for better hover display
        filtered_df['hover_title'] = filtered_df['title'].apply(
            lambda x: '<br>'.join(textwrap.wrap(x, width=40))
        )
        filtered_df['hover_text'] = (
            filtered_df['hover_title'] + '<br>' + 
            filtered_df['published'].dt.strftime('%Y-%m-%d %H:%M') + 
            '<br>Length: ' + filtered_df['doc_length'].astype(str) + ' tokens<br>' +
            'Topic: ' + filtered_df['dominant_topic'].astype(str)
        )

        fig = px.scatter(
            filtered_df,
            x='published',
            y='doc_length',
            size='doc_length',
            color='dominant_topic',
            size_max=30,
            hover_data=['hover_text'],
            title='Document Length Over Time',
            labels={
                'published': 'Publication Date',
                'doc_length': 'Article Length (tokens)',
                'dominant_topic': 'Topic'
            },
            log_y=True
        )
        
        fig.update_traces(
            hoverinfo="text",
            hovertemplate="%{customdata[0]}",
            marker=dict(
                opacity=0.8,
                line=dict(width=0.5, color='white' if dark_mode else 'black')
            )
        )
        
        fig.update_layout(**get_guardian_plot_layout('', dark_mode))
        fig.update_layout(
            xaxis=dict(
                title='Publication Date',
                tickformat='%d %b %Y',
                tickmode='auto',
                nticks=10
            ),
            yaxis=dict(
                title='Article Length (tokens)',
                type='log'
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating bubble chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating bubble chart: {e}", dark_mode))
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
            fig.update_layout(**get_guardian_plot_layout("No bigrams/trigrams found", dark_mode))
            return fig

        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        top_ngrams = sorted_ngrams[:10]
        
        # Format ngrams for better readability
        formatted_ngrams = []
        for ngram, count in top_ngrams:
            # Replace underscores with spaces
            formatted = ngram.replace('_', ' ')
            formatted_ngrams.append((formatted, count))
        
        df_ngram = pd.DataFrame(formatted_ngrams, columns=["ngram", "count"])

        fig = px.line_polar(
            df_ngram,
            r="count",
            theta="ngram",
            line_close=True,
            title="Top Bigrams & Trigrams",
            color_discrete_sequence=[GUARDIAN_COLORS["blue"]]
        )
        fig.update_traces(
            fill='toself',
            fillcolor=f"rgba({0},{86},{137},{0.3})"  # Semi-transparent guardian blue
        )
        
        fig.update_layout(**get_guardian_plot_layout('', dark_mode))
        # Enhance the polar layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(df_ngram["count"]) * 1.1]
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=False
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating ngram radar chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating ngram radar chart: {e}", dark_mode))
        return fig

def create_topic_network(topic_similarity, lda_model, dark_mode=False):
    """
    Create a network visualization of topic relationships based on similarity.
    """
    try:
        if not topic_similarity:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("No topic similarity data available", dark_mode))
            return fig
            
        # Create a network graph
        G = nx.Graph()
        
        # Add nodes (topics)
        for topic_id in range(lda_model.num_topics):
            # Get top words for the topic for node labels
            top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=3)]
            topic_label = f"Topic {topic_id}: {', '.join(top_words)}"
            G.add_node(topic_id, label=topic_label)
        
        # Add edges (relationships between topics)
        for topic1, relations in topic_similarity.items():
            for topic2, strength in relations.items():
                if strength > 0.2:  # Only add edges for sufficiently strong relationships
                    G.add_edge(int(topic1), int(topic2), weight=strength)
        
        # Use spring layout to position nodes
        pos = nx.spring_layout(G, seed=42)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G.edges[edge]['weight']
            
            # Adjust line width based on weight
            width = weight * 5
            
            edge_trace = go.Scatter(
                x=[x0, x1], 
                y=[y0, y1],
                line=dict(width=width, color='rgba(150,150,150,0.3)'),
                hoverinfo='text',
                text=f"Similarity: {weight:.2f}",
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            # Size based on degree (number of connections)
            node_size.append(10 + G.degree(node) * 5)
            # Color based on topic ID
            node_color.append(node)
        
        node_trace = go.Scatter(
            x=node_x, 
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                line=dict(width=2, color='black' if not dark_mode else 'white')
            ),
            text=[f"Topic {i}" for i in range(len(node_text))],
            textposition="bottom center",
            hoverinfo='text',
            hovertext=node_text
        )
        
        # Create the figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        fig.update_layout(
            title="Topic Similarity Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            **get_guardian_plot_layout('', dark_mode)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating topic network: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating topic network: {e}", dark_mode))
        return fig

def create_coherence_chart(coherence_data, dark_mode=False):
    """
    Create a bar chart showing topic coherence scores.
    """
    try:
        if not coherence_data:
            fig = go.Figure()
            fig.update_layout(**get_guardian_plot_layout("No coherence data available", dark_mode))
            return fig
            
        # Create dataframe for plotting
        df = pd.DataFrame(coherence_data)
        
        # Sort by coherence
        df = df.sort_values('coherence', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            df,
            x='topic',
            y='coherence',
            color='coherence',
            title="Topic Coherence Scores",
            labels={'topic': 'Topic', 'coherence': 'Coherence Score'},
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig.update_layout(**get_guardian_plot_layout('', dark_mode))
        fig.update_layout(
            xaxis_title="Topic",
            yaxis_title="Coherence Score (higher is better)",
            coloraxis_showscale=False,
            hovermode="closest"
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="Topic %{x}<br>Coherence: %{y:.3f}<extra></extra>"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating coherence chart: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(**get_guardian_plot_layout(f"Error creating coherence chart: {e}", dark_mode))
        return fig

# ─────────────────────────────────────────────────────────────────────
# Custom components
# ─────────────────────────────────────────────────────────────────────
def create_collapsible_card(header, content, card_id, is_open=True, className="mb-3", icon=None):
    """Create a collapsible Bootstrap card"""
    header_content = [
        html.Span(header, className="me-auto"),
        html.I(className=f"fas fa-chevron-down ms-2 collapse-icon {'expanded' if is_open else ''}"),
    ]
    
    if icon:
        header_content.insert(0, html.I(className=f"fas fa-{icon} me-2"))
    
    return dbc.Card([
        dbc.CardHeader(
            html.Div(header_content, className="d-flex align-items-center"),
            id=f"{card_id}-header",
            className="d-flex justify-content-between align-items-center"
        ),
        dbc.Collapse(
            dbc.CardBody(content),
            id=f"{card_id}-collapse",
            is_open=is_open
        )
    ], className=f"collapsible-card {className}")

def create_tooltip(text, tooltip_text):
    """Create a tooltip with help text"""
    return html.Span([
        text,
        html.Span(
            [
                html.I(className="fas fa-info-circle ms-2"),
                html.Span(tooltip_text, className="tooltiptext")
            ],
            className="guardian-tooltip"
        )
    ])

def create_topic_button(topic_id, is_selected=False):
    """Create a topic selection button"""
    button_class = (
        "btn btn-sm topic-btn "
        f"{'btn-primary' if is_selected else 'btn-outline-secondary'}"
    )
    return html.Button(
        f"Topic {topic_id}",
        id=f"topic-btn-{topic_id}",
        className=button_class,
        n_clicks=0
    )

# ─────────────────────────────────────────────────────────────────────
# Layout Components
# ─────────────────────────────────────────────────────────────────────
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src="https://static.guim.co.uk/sys-images/Guardian/Pix/pictures/2010/03/01/poweredbyguardianBLACK.png", height="30px"), width="auto"),
                dbc.Col(dbc.NavbarBrand("Guardian News Topic Explorer", className="ms-2")),
            ], align="center", className="g-0"),
            href="#",
            style={"textDecoration": "none"},
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("About", href="#about")),
                dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model", target="_blank")),
                dbc.NavItem(
                    dbc.Button("Toggle Dark Mode", id="dark-mode-toggle", color="light", outline=True, size="sm", className="ms-2")
                ),
            ], className="ms-auto"),
            id="navbar-collapse",
            is_open=False,
            navbar=True
        ),
    ]),
    color=GUARDIAN_COLORS["blue"],
    dark=True,
    className="mb-4"
)

about_card = create_collapsible_card(
    header="About This Dashboard",
    content=html.P([
        """This dashboard fetches articles from The Guardian's API, 
        processes them with Natural Language Processing (LDA topic modeling, 
        bigrams/trigrams detection), and visualizes them through various plots. 
        Explore how news topics emerge and evolve over time, discover hidden topics 
        in the articles, and see how they relate to each other. Use the controls 
        below to customize your analysis.""",
        html.A(
            "Learn more on GitHub",
            href="https://github.com/StephenJudeD/Guardian-News-RSS---LDA-Model",
            target="_blank",
            className="ms-1"
        )
    ]),
    card_id="about",
    icon="info-circle"
)

date_controls = dbc.Card([
    dbc.CardHeader(create_tooltip(
        "Date Range Selection", 
        "Select the time period for articles to analyze"
    )),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Last Day", id="date-1d", n_clicks=0, color="outline-primary", size="sm"),
                    dbc.Button("Last 3 Days", id="date-3d", n_clicks=0, color="outline-primary", size="sm"),
                    dbc.Button("Last Week", id="date-7d", n_clicks=0, color="primary", size="sm"),
                    dbc.Button("Last Month", id="date-30d", n_clicks=0, color="outline-primary", size="sm"),
                ], className="w-100 mb-3"),
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
    ])
], className="mb-3 hover-card")

topic_controls = dbc.Card([
    dbc.CardHeader(create_tooltip(
        "LDA Topic Settings", 
        "Configure how many topics the algorithm should find in the articles"
    )),
    dbc.CardBody([
        dbc.Label("Number of Topics", html_for="num-topics-slider"),
        dcc.Slider(
            id="num-topics-slider",
            min=2,
            max=15,
            step=1,
            value=5,
            marks={i: str(i) for i in range(2, 16, 2)},
            tooltip={"placement": "bottom", "always_visible": True},
            className="mb-3"
        ),
        html.Div([
            html.Small("More topics → more specific categories, but may be less coherent.", className="text-muted"),
            html.Small("Fewer topics → broader categories with better coherence.", className="text-muted d-block"),
        ])
    ])
], className="mb-3 hover-card")

tsne_controls = dbc.Card([
    dbc.CardHeader(create_tooltip(
        "t-SNE Visualization Settings", 
        "Configure the t-SNE algorithm that creates the 3D topic visualization"
    )),
    dbc.CardBody([
        dbc.Label("Perplexity", html_for="tsne-perplexity-slider"),
        dcc.Slider(
            id="tsne-perplexity-slider",
            min=5,
            max=50,
            step=5,
            value=30,
            marks={i: str(i) for i in range(5, 51, 10)},
            tooltip={"placement": "bottom", "always_visible": True},
            className="mb-3"
        ),
        html.Div([
            html.Small("Lower perplexity → focus on local structure, better for small datasets", className="text-muted"),
            html.Small("Higher perplexity → maintain global structure, better for large datasets", className="text-muted d-block"),
        ])
    ])
], className="mb-3 hover-card")

update_button = dbc.Row([
    dbc.Col([
        dbc.Button(
            [html.I(className="fas fa-sync-alt me-2"), "Update Analysis"],
            id="update-button",
            color="primary",
            size="lg",
            className="w-100"
        ),
    ], width={"size": 6, "offset": 3}),
], className="mb-4 mt-3")

topic_selector = dbc.Card([
    dbc.CardHeader(create_tooltip(
        "Topic Explorer", 
        "Select a topic to view its word cloud and related articles"
    )),
    dbc.CardBody([
        html.Div(id="topic-buttons", className="mb-3"),
        html.P(
            "Select a topic above to focus the word cloud and filter articles by that topic.", 
            className="text-muted small"
        )
    ])
], className="mb-3 d-none", id="topic-selector-card")

topic_distribution_card = create_collapsible_card(
    header=create_tooltip(
        "Topic Word Distributions", 
        "Shows the top words for each topic and their probabilities"
    ),
    content=dcc.Loading(
        dcc.Graph(id="topic-distribution", config={'displayModeBar': True, 'scrollZoom': True}),
        type="circle"
    ),
    card_id="topic-dist",
    className="mb-3"
)

word_cloud_card = create_collapsible_card(
    header=create_tooltip(
        "Topic Word Cloud", 
        "Visual representation of the most important words in the selected topic"
    ),
    content=dcc.Loading(
        dcc.Graph(id="word-cloud", config={'displayModeBar': False}),
        type="circle"
    ),
    card_id="word-cloud",
    className="mb-3"
)

tsne_card = create_collapsible_card(
    header=create_tooltip(
        "3D Topic Clustering (t-SNE)", 
        "Interactive 3D visualization showing how articles cluster by topic similarity"
    ),
    content=dcc.Loading(
        dcc.Graph(id="tsne-plot", config={'displayModeBar': True}),
        type="circle"
    ),
    card_id="tsne",
    className="mb-3"
)

bubble_chart_card = create_collapsible_card(
    header=create_tooltip(
        "Document Length Over Time", 
        "Bubble chart showing article length over time, colored by dominant topic"
    ),
    content=dcc.Loading(
        dcc.Graph(id="bubble-chart", config={'displayModeBar': True}),
        type="circle"
    ),
    card_id="bubble",
    className="mb-3"
)

ngram_chart_card = create_collapsible_card(
    header=create_tooltip(
        "Bigrams & Trigrams", 
        "Most frequent multi-word phrases in the articles"
    ),
    content=dcc.Loading(
        dcc.Graph(id="ngram-chart", config={'displayModeBar': True}),
        type="circle"
    ),
    card_id="ngram",
    className="mb-3"
)

topic_network_card = create_collapsible_card(
    header=create_tooltip(
        "Topic Network", 
        "Network visualization showing relationships between topics"
    ),
    content=dcc.Loading(
        dcc.Graph(id="topic-network", config={'displayModeBar': True}),
        type="circle"
    ),
    card_id="topic-network",
    className="mb-3"
)

coherence_card = create_collapsible_card(
    header=create_tooltip(
        "Topic Coherence Metrics", 
        "Measures how semantically coherent each topic is (higher is better)"
    ),
    content=dcc.Loading(
        dcc.Graph(id="coherence-chart", config={'displayModeBar': True}),
        type="circle"
    ),
    card_id="coherence",
    className="mb-3"
)

article_table_card = create_collapsible_card(
    header="Article Details",
    content=[
        dbc.Row([
            dbc.Col(html.Div(id="article-count", className="mb-3"), width=8),
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
                ], size="sm")
            ], width=4)
        ]),
        dash_table.DataTable(
            id="article-table",
            columns=[
                {"name": "Title", "id": "title", "presentation": "markdown"},
                {"name": "Published", "id": "published"},
                {"name": "Topics", "id": "topics", "presentation": "markdown"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "maxWidth": 0,
                "fontSize": 14,
                "fontFamily": "Georgia, serif",
                "padding": "10px"
            },
            style_header={
                "backgroundColor": GUARDIAN_COLORS["blue"],
                "color": "white",
                "fontWeight": "bold",
                "fontSize": 15
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "rgba(0, 0, 0, 0.05)"
                }
            ],
            markdown_options={"html": True},
            page_size=10,
            page_action="native",
            sort_action="native",
            filter_action="native",
            style_as_list_view=True,
        ),
        html.Div(id="article-pagination", className="mt-3 text-center")
    ],
    card_id="articles",
    className="mb-3"
)

app.layout = html.Div([
    dcc.Store(id="app-data"),
    dcc.Store(id="dark-mode-store", data=False),
    # Dark mode script and style
    html.Script(dark_mode_script),
    
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col([
                about_card,
                dbc.Row([
                    dbc.Col(date_controls, md=4),
                    dbc.Col(topic_controls, md=4),
                    dbc.Col(tsne_controls, md=4),
                ]),
                update_button,
                topic_selector,
                
                # Main content area
                dbc.Tabs([
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col(topic_distribution_card, lg=6),
                            dbc.Col(word_cloud_card, lg=6),
                        ]),
                        dbc.Row([
                            dbc.Col(tsne_card, lg=6),
                            dbc.Col(bubble_chart_card, lg=6),
                        ]),
                    ], label="Main Visualizations", tab_id="tab-main"),
                    
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col(ngram_chart_card, lg=6),
                            dbc.Col(topic_network_card, lg=6),
                        ]),
                        dbc.Row([
                            dbc.Col(coherence_card),
                        ]),
                    ], label="Advanced Analysis", tab_id="tab-advanced"),
                    
                    dbc.Tab([
                        article_table_card
                    ], label="Article Details", tab_id="tab-articles"),
                ], id="main-tabs"),
                
                html.Footer([
                    html.Hr(),
                    html.P([
                        "Data sourced from ",
                        html.A("The Guardian Open Platform", href="https://open-platform.theguardian.com/", target="_blank"),
                        ". This dashboard is for educational purposes only."
                    ], className="text-center text-muted mt-4 mb-4")
                ])
            ])
        ])
    ], fluid=True)
])

# ─────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────

# Dark mode toggle
@app.callback(
    Output("dark-mode-store", "data"),
    Input("dark-mode-toggle", "n_clicks"),
    State("dark-mode-store", "data"),
)
def toggle_dark_mode(n_clicks, current_mode):
    if n_clicks is None:
        return current_mode
    return not current_mode

# Date range button callbacks
@app.callback(
    [Output("start-date", "value"), Output("end-date", "value")],
    [
        Input("date-1d", "n_clicks"),
        Input("date-3d", "n_clicks"),
        Input("date-7d", "n_clicks"),
        Input("date-30d", "n_clicks"),
    ],
)
def update_date_range(n1, n3, n7, n30):
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
    elif button_id == "date-30d":
        start_date = end_date - timedelta(days=30)
    else:
        # Default
        start_date = end_date - timedelta(days=7)
        
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Button styling for date range
@app.callback(
    [
        Output("date-1d", "color"),
        Output("date-3d", "color"),
        Output("date-7d", "color"),
        Output("date-30d", "color"),
    ],
    [
        Input("date-1d", "n_clicks"),
        Input("date-3d", "n_clicks"),
        Input("date-7d", "n_clicks"),
        Input("date-30d", "n_clicks"),
    ],
)
def update_date_button_style(n1, n3, n7, n30):
    ctx = callback_context
    if not ctx.triggered:
        # Default to 7 days
        return "outline-primary", "outline-primary", "primary", "outline-primary"
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "date-1d":
        return "primary", "outline-primary", "outline-primary", "outline-primary"
    elif button_id == "date-3d":
        return "outline-primary", "primary", "outline-primary", "outline-primary"
    elif button_id == "date-7d":
        return "outline-primary", "outline-primary", "primary", "outline-primary"
    elif button_id == "date-30d":
        return "outline-primary", "outline-primary", "outline-primary", "primary"
    else:
        # Default
        return "outline-primary", "outline-primary", "primary", "outline-primary"

# Collapsible card callbacks
for section in ["about", "topic-dist", "word-cloud", "tsne", "bubble", "ngram", "topic-network", "coherence", "articles"]:
    @app.callback(
        [Output(f"{section}-collapse", "is_open"),
         Output(f"{section}-header", "children")],
        [Input(f"{section}-header", "n_clicks")],
        [State(f"{section}-collapse", "is_open"),
         State(f"{section}-header", "children")]
    )
    def toggle_collapse(n, is_open, header):
        if n:
            # Toggle the collapse state
            # Update the chevron icon direction
            header[1].className = f"fas fa-chevron-down ms-2 collapse-icon {'expanded' if not is_open else ''}"
            return not is_open, header
        return is_open, header

# Main data processing callback
@app.callback(
    [Output("app-data", "data"),
     Output("topic-buttons", "children"),
     Output("topic-selector-card", "className"),
     Output("topic-filter", "options"),
     Output("article-count", "children")],
    [Input("update-button", "n_clicks")],
    [State("start-date", "value"),
     State("end-date", "value"),
     State("num-topics-slider", "value"),
     State("tsne-perplexity-slider", "value")]
)
def process_and_store_data(n_clicks, start_date, end_date, num_topics, perplexity):
    if n_clicks is None:
        # Initial load, return empty data
        return None, [], "mb-3 d-none", [{"label": "All Topics", "value": "all"}], ""
        
    # Process articles
    df, texts, dictionary, corpus, lda_model = process_articles(start_date, end_date, num_topics)
    
    if df is None or df.empty:
        return None, [], "mb-3 d-none", [{"label": "All Topics", "value": "all"}], "No articles found for the selected date range."
    
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
    
    # Calculate coherence metrics
    coherence_metrics = calculate_coherence(texts, dictionary, lda_model)
    
    # Calculate topic similarity
    topic_similarities = calculate_topic_similarity(lda_model)
    
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
        doc_topics, _, _ = lda_model[corpus[i]]  # Unpack all three values
        for topic_id, w in doc_topics:  # Now iterate over the topic distribution
            topic_weights[topic_id] = w
        doc_topics_list.append(topic_weights)
    
    doc_topics_array = np.array(doc_topics_list, dtype=np.float32)
    
    perplex_val = min(perplexity, max(2, len(doc_topics_array) - 1))
    tsne = TSNE(
        n_components=3,
        random_state=42,
        perplexity=perplex_val,
        n_jobs=-1,
        learning_rate='auto'
    )
    embedded = tsne.fit_transform(doc_topics_array)
    
    tsne_data = {
        "x": embedded[:, 0].tolist(),
        "y": embedded[:, 1].tolist(),
        "z": embedded[:, 2].tolist(),
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
        "coherence_metrics": coherence_metrics,
        "topic_similarities": topic_similarities,
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
            className="btn btn-sm btn-primary topic-btn",
            n_clicks=0
        )
    ]
    
    for i in range(num_topics):
        buttons.append(
            html.Button(
                f"Topic {i}",
                id=f"topic-btn-{i}",
                className="btn btn-sm btn-outline-secondary topic-btn",
                n_clicks=0
            )
        )
    
    # Create topic filter options
    filter_options = [{"label": "All Topics", "value": "all"}]
    for i in range(num_topics):
        filter_options.append({"label": f"Topic {i}", "value": str(i)})
    
    # Show topic selector
    selector_class = "mb-3"
    
    # Article count message
    count_message = html.Div([
        html.Strong(f"Found {len(articles)} articles"),
        html.Span(f" from {start_date} to {end_date}")
    ])
    
    return data, buttons, selector_class, filter_options, count_message

# Topic button callbacks - dynamically generate callbacks for each possible topic button
@app.callback(
    [Output(f"topic-btn-all", "className")] +
    [Output(f"topic-btn-{i}", "className") for i in range(15)],  # Support up to 15 topics
    [Input(f"topic-btn-all", "n_clicks")] +
    [Input(f"topic-btn-{i}", "n_clicks") for i in range(15)],
    [State("app-data", "data"), State("topic-filter", "value")]
)
def update_active_topic(*args):
    # Extract n_clicks and data from args
    n_clicks_list = args[:16]  # First 16 args are n_clicks
    data = args[16]  # Next arg is app-data
    current_filter = args[17]  # Last arg is topic-filter value
    
    ctx = callback_context
    if not ctx.triggered or data is None:
        # Initial load, keep "All Topics" selected
        return ["btn btn-sm btn-primary topic-btn"] + ["btn btn-sm btn-outline-secondary topic-btn"] * 15
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    topic_id = button_id.split('-')[-1]  # Extract topic id from button id
    
    # Update active button
    result = ["btn btn-sm btn-outline-secondary topic-btn"] * 16
    if topic_id == "all":
        result[0] = "btn btn-sm btn-primary topic-btn"
    else:
        try:
            index = int(topic_id) + 1  # +1 because "all" is at index 0
            result[index] = "btn btn-sm btn-primary topic-btn"
        except:
            # Default to "all" if something goes wrong
            result[0] = "btn btn-sm btn-primary topic-btn"
    
    return result

# Update visualizations based on app data
@app.callback(
    [Output("topic-distribution", "figure"),
     Output("word-cloud", "figure"),
     Output("tsne-plot", "figure"),
     Output("bubble-chart", "figure"),
     Output("ngram-chart", "figure"),
     Output("topic-network", "figure"),
     Output("coherence-chart", "figure"),
     Output("article-table", "data"),
     Output("topic-filter", "value")],
    [Input("app-data", "data"),
     Input("topic-btn-all", "n_clicks")] +
    [Input(f"topic-btn-{i}", "n_clicks") for i in range(15)] +  # Support up to 15 topics
    [Input("topic-filter", "value"),
     Input("dark-mode-store", "data")]
)
def update_visualizations(data, *args):
    # Extract n_clicks, filter value, and dark mode from args
    n_clicks_list = args[:16]  # First 16 args are topic button n_clicks
    filter_value = args[16]  # Next arg is topic-filter value
    dark_mode = args[17]  # Last arg is dark-mode-store value
    
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Determine selected topic
    selected_topic = "all"
    if triggered_id and triggered_id.startswith("topic-btn-"):
        selected_topic = triggered_id.split('-')[-1]
    elif triggered_id == "topic-filter":
        selected_topic = filter_value
    
    # Default empty figures
    empty_fig = go.Figure().update_layout(**get_guardian_plot_layout("No data available", dark_mode))
    
    if data is None:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], "all"
    
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
        
        topic_dist_fig.update_layout(**get_guardian_plot_layout("", dark_mode))
        topic_dist_fig.update_layout(
            yaxis=dict(autorange="reversed"),
            hovermode="closest"
        )
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
            word_cloud_fig = create_word_cloud(
                data["word_clouds"][str(word_cloud_topic)], 
                dark_mode
            )
        else:
            word_cloud_fig = empty_fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}", exc_info=True)
        word_cloud_fig = empty_fig
    
    # t-SNE visualization
    try:
        tsne_fig = go.Figure(data=[
            go.Scatter3d(
                x=data["tsne_data"]["x"],
                y=data["tsne_data"]["y"],
                z=data["tsne_data"]["z"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=data["tsne_data"]["topic"],
                    colorscale="Viridis",
                    opacity=0.8,
                    line=dict(
                        width=0.5,
                        color="white" if dark_mode else "black"
                    )
                ),
                text=data["tsne_data"]["title"],
                hoverinfo="text"
            )
        ])
        
        tsne_fig.update_layout(**get_guardian_plot_layout("3D t-SNE Topic Clustering", dark_mode))
        tsne_fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(showticklabels=False, title=""),
                annotations=[dict(
                    showarrow=False,
                    x=0.05,
                    y=0.05,
                    z=0.05,
                    text="Drag to rotate, scroll to zoom",
                    font=dict(color="white" if dark_mode else "black")
                )]
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
                labels={
                    "published": "Publication Date",
                    "doc_length": "Article Length (tokens)",
                    "dominant_topic": "Dominant Topic"
                }
            )
            
            bubble_fig.update_layout(**get_guardian_plot_layout("Document Length Over Time", dark_mode))
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
    
    # Ngram chart
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
            fill="toself",
            fillcolor=f"rgba({0},{86},{137},{0.3})"  # Semi-transparent guardian blue
        )
        
        ngram_fig.update_layout(**get_guardian_plot_layout("", dark_mode))
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
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (topics)
            for topic_id in range(data["num_topics"]):
                # Get top words for the topic for node labels
                top_words = [word for word, _ in data["topic_distributions"][str(topic_id)]][:3]
                topic_label = f"Topic {topic_id}: {', '.join(top_words)}"
                G.add_node(topic_id, label=topic_label)
            
            # Add edges (relationships between topics)
            for topic1, relations in data["topic_similarities"].items():
                for topic2, strength in relations.items():
                    if float(strength) > 0.2:  # Only add edges for sufficiently strong relationships
                        G.add_edge(int(topic1), int(topic2), weight=float(strength))
            
            # Use spring layout to position nodes
            pos = nx.spring_layout(G, seed=42)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = G.edges[edge]['weight']
                
                # Adjust line width based on weight
                width = weight * 5
                
                edge_trace = go.Scatter(
                    x=[x0, x1], 
                    y=[y0, y1],
                    line=dict(
                        width=width, 
                        color='rgba(150,150,150,0.3)' if dark_mode else 'rgba(100,100,100,0.3)'
                    ),
                    hoverinfo='text',
                    text=f"Similarity: {weight:.2f}",
                    mode='lines'
                )
                edge_traces.append(edge_trace)
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(G.nodes[node]['label'])
                # Size based on degree (number of connections)
                node_size.append(10 + G.degree(node) * 5)
                # Color based on topic ID
                node_color.append(node)
            
            node_trace = go.Scatter(
                x=node_x, 
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale='Viridis',
                    line=dict(width=2, color='white' if dark_mode else 'black')
                ),
                text=[f"Topic {i}" for i in range(len(node_text))],
                textposition="bottom center",
                hoverinfo='text',
                hovertext=node_text
            )
            
            # Create the figure
            network_fig = go.Figure(data=edge_traces + [node_trace])
            
            # Update layout
            network_fig.update_layout(
                title="Topic Similarity Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                **get_guardian_plot_layout('', dark_mode)
            )
        else:
            network_fig = empty_fig
    except Exception as e:
        logger.error(f"Error creating topic network: {e}", exc_info=True)
        network_fig = empty_fig
    
    # Coherence metrics chart
    try:
        if "coherence_metrics" in data:
            coherence_df = pd.DataFrame(data["coherence_metrics"])
            
            # Sort by coherence
            coherence_df = coherence_df.sort_values('coherence', ascending=False)
            
            # Create bar chart
            coherence_fig = px.bar(
                coherence_df,
                x='topic',
                y='coherence',
                color='coherence',
                title="Topic Coherence Scores",
                labels={'topic': 'Topic', 'coherence': 'Coherence Score'},
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            coherence_fig.update_layout(**get_guardian_plot_layout('', dark_mode))
            coherence_fig.update_layout(
                xaxis_title="Topic",
                yaxis_title="Coherence Score (higher is better)",
                coloraxis_showscale=False,
                hovermode="closest"
            )
            
            # Update hover template
            coherence_fig.update_traces(
                hovertemplate="Topic %{x}<br>Coherence: %{y:.3f}<extra></extra>"
            )
        else:
            coherence_fig = empty_fig
    except Exception as e:
        logger.error(f"Error creating coherence chart: {e}", exc_info=True)
        coherence_fig = empty_fig
    
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
        network_fig,
        coherence_fig,
        articles,
        new_filter_value
    )

# Navbar toggle callback
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=False)
