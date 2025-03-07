import dash
from dash import Dash, dcc, html, Input, Output, dash_table, State
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
from openai import OpenAI

# ===========================================
# INITIALIZATION & CONFIGURATION
# ===========================================
load_dotenv()
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===========================================
# VISUAL IDENTITY SYSTEM
# ===========================================
DARK_GRAPHITE = "#1a1a1a"
ELECTRIC_BLUE = "#00F3FF"
NEON_PINK = "#FF10F0"
TOPIC_COLORS = {
    0: "#2196F3",  # Vivid Blue
    1: "#FF5252",  # Vivid Red
    2: "#9C27B0",  # Vivid Purple
    3: "#4CAF50",  # Vivid Green
    4: "#FF9800"   # Vivid Orange
}

PLOTLY_THEME = {
    "layout": {
        "template": "plotly_dark",
        "paper_bgcolor": DARK_GRAPHITE,
        "plot_bgcolor": DARK_GRAPHITE,
        "font": {"family": "Arial", "color": ELECTRIC_BLUE},
        "xaxis": {
            "gridcolor": "rgba(0, 243, 255, 0.2)",
            "linecolor": ELECTRIC_BLUE,
            "title_font": {"size": 14, "color": NEON_PINK}
        },
        "yaxis": {
            "gridcolor": "rgba(0, 243, 255, 0.2)",
            "linecolor": ELECTRIC_BLUE,
            "title_font": {"size": 14, "color": NEON_PINK}
        },
        "hoverlabel": {
            "bgcolor": DARK_GRAPHITE,
            "font_size": 12,
            "font_color": ELECTRIC_BLUE
        },
        "colorway": list(TOPIC_COLORS.values()),
        "margin": {"l": 50, "r": 50, "t": 80, "b": 50},
        "transition": {"duration": 300}
    }
}

# ===========================================
# CORE APPLICATION
# ===========================================
app = Dash(__name__, 
          external_stylesheets=[dbc.themes.DARKLY],
          suppress_callback_exceptions=True)
server = app.server

guardian = GuardianFetcher(os.getenv('GUARDIAN_API_KEY'))
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
stop_words = set(stopwords.words('english')).union({
    'said', 'would', 'also', 'year', 'new', 'like', 'us', 'people'
})

# ===========================================
# DATA PROCESSING
# ===========================================
@lru_cache(maxsize=4)
def process_articles(days_back=7):
    """Process articles with caching and phrase detection"""
    try:
        logger.info(f"Processing articles from last {days_back} days")
        df = guardian.fetch_articles(days_back=days_back, page_size=100)
        if df.empty:
            return None, None, None, None, None, None

        # Enhanced text processing with phrase detection
        tokenized_texts = []
        for content in df['content']:
            words = word_tokenize(str(content).lower())
            filtered = [w for w in words if w.isalnum() and w not in stop_words]
            tokenized_texts.append(filtered)

        # Phrase detection with proper combining
        bigram = Phrases(tokenized_texts, min_count=5, threshold=10)
        trigram = Phrases(bigram[tokenized_texts], threshold=10)
        texts = [
            ['_'.join(phrase) for phrase in trigram[bigram[doc]] 
             if len(phrase) > 3 and not any(c.isdigit() for c in phrase)]
            for doc in tokenized_texts
        ]

        # LDA Model training
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=5,
            id2word=dictionary,
            passes=4,
            random_state=42,
            chunksize=100,
            alpha='auto'
        )

        # Improved topic naming with OpenAI
        topic_names = {}
        for topic_id in range(lda_model.num_topics):
            top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=50)]
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": f"""Generate a concise 2-3 word news topic modelling name  based on these key phrases: {', '.join(top_words[:10])}.
                        Examples of good names: 'Climate Policy', 'Tech Innovation', 'Foreign Relations', 'Political Affairs', 'Sport', 'Film & Culture', 
                        Bad examples: 'Various Topics', 'Mixed Issues', 'General News', using quatation marks
                        Topic Name:"""
                    }]
                )
                topic_names[topic_id] = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                topic_names[topic_id] = f"Topic {topic_id}"

        return df, texts, dictionary, corpus, lda_model, topic_names

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return None, None, None, None, None, None

# ===========================================
# VISUALIZATION COMPONENTS
# ===========================================
def create_topic_nexus(lda_model, topic_names):
    """Full-width radar chart with phrases"""
    fig = go.Figure()
    for topic_id in range(lda_model.num_topics):
        words, probs = zip(*lda_model.show_topic(topic_id, topn=15))
        fig.add_trace(go.Scatterpolar(
            r=probs + probs[:1],
            theta=words + words[:1],
            fill='toself',
            name=topic_names[topic_id],
            line_color=TOPIC_COLORS[topic_id],
            opacity=0.8
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                showline=False, 
                gridcolor='rgba(0, 243, 255, 0.2)',
                range=[0, max(probs)*1.1]
            ),
            angularaxis=dict(
                linecolor=ELECTRIC_BLUE, 
                gridcolor='rgba(0, 243, 255, 0.2)',
                rotation=45
            )
        ),
        title={
            'text': "<b>TOPIC WORD RADAR</b>",
            'font': {'size': 24, 'color': NEON_PINK},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600,
        margin={"t": 100}
    )
    return fig

def create_tsne_3d_visualization(df, lda_model, corpus, topic_names):
    """Interactive 3D topic clustering"""
    try:
        topic_weights = np.array([[weight for (_, weight) in lda_model[doc]] for doc in corpus])
        
        tsne = TSNE(n_components=3, perplexity=min(30, len(df)-1), random_state=42)
        embeddings = tsne.fit_transform(topic_weights)
        
        df_vis = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'z': embeddings[:, 2],
            'topic': [np.argmax(row) for row in topic_weights],
            'title': df['title'],
            'date': df['published'].dt.strftime('%Y-%m-%d')
        })
        
        fig = px.scatter_3d(
            df_vis,
            x='x', y='y', z='z',
            color='topic',
            color_discrete_map=TOPIC_COLORS,
            hover_data=['title', 'date'],
            title="<b>3D ARTICLE CLUSTER MAP</b>"
        )
        
        fig.update_layout(
            title={'x': 0.5, 'font': {'size': 24, 'color': NEON_PINK}},
            scene=dict(
                xaxis=dict(backgroundcolor=DARK_GRAPHITE),
                yaxis=dict(backgroundcolor=DARK_GRAPHITE),
                zaxis=dict(backgroundcolor=DARK_GRAPHITE)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig
    except Exception as e:
        logger.error(f"3D TSNE error: {e}")
        return go.Figure()

# ===========================================
# APP LAYOUT
# ===========================================
app.layout = dbc.Container([
    # Header
    dbc.Navbar(
        dbc.Container([
            html.Img(src="", height="40px", className="me-2"),
            dbc.NavbarBrand(
                "GUARDIAN TOPIC EXPLORER",
                style={"fontSize": "2rem", "letterSpacing": "2px"}
            )
        ]),
        color=DARK_GRAPHITE,
        dark=True,
        className="mb-4"
    ),
    
    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("TIME RANGE", className="text-center"),
                dbc.CardBody([
                    dbc.RadioItems(
                        id='time-range',
                        options=[
                            {'label': '24 Hours', 'value': 1},
                            {'label': '3 Days', 'value': 3},
                            {'label': '7 Days', 'value': 7}
                        ],
                        value=7,
                        inline=True
                    )
                ])
            ], className="shadow-lg")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("TOPIC FILTER", className="text-center"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='topic-filter',
                        multi=True,
                        placeholder="Select topics...",
                        style={'minWidth': '200px'}
                    )
                ])
            ], className="shadow-lg")
        ], md=6)
    ], className="g-4 mb-4"),
    
    # Visualizations
    dbc.Row([
        dbc.Col(dcc.Graph(id='topic-nexus'), width=12)
    ], className="g-4 mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='tsne-3d-plot'), width=12)
    ], className="g-4 mb-4"),
    
    # Data Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("ARTICLE DETAILS", className="text-center"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='article-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'backgroundColor': DARK_GRAPHITE,
                            'color': ELECTRIC_BLUE,
                            'border': f'1px solid {ELECTRIC_BLUE}33'
                        },
                        style_header={
                            'backgroundColor': DARK_GRAPHITE,
                            'fontWeight': 'bold',
                            'border': f'1px solid {ELECTRIC_BLUE}'
                        },
                        page_size=10
                    )
                ])
            ], className="shadow-lg")
        ])
    ])
], fluid=True, style={"backgroundColor": DARK_GRAPHITE})

# ===========================================
# CALLBACKS
# ===========================================
@app.callback(
    [Output('topic-nexus', 'figure'),
     Output('tsne-3d-plot', 'figure'),
     Output('article-table', 'data'),
     Output('topic-filter', 'options')],
    [Input('time-range', 'value')],
    [State('topic-filter', 'value')]
)
def update_all_visuals(days_back, selected_topics):
    df, texts, dictionary, corpus, lda_model, topic_names = process_articles(days_back)
    if df is None:
        return go.Figure(), go.Figure(), [], []
    
    # Generate visualizations
    nexus_fig = create_topic_nexus(lda_model, topic_names)
    tsne_3d_fig = create_tsne_3d_visualization(df, lda_model, corpus, topic_names)
    
    # Prepare table data
    df['top_words'] = df.apply(lambda row: ', '.join(
        [word for word, _ in lda_model.show_topic(
            np.argmax([prob for _, prob in lda_model[corpus[row.name]]]), 
            topn=5
        )]
    ), axis=1)
    
    table_data = df[['title', 'published', 'section', 'top_words']].to_dict('records')
    
    # Update filter options
    topic_options = [{'label': name, 'value': tid} for tid, name in topic_names.items()]
    
    return nexus_fig, tsne_3d_fig, table_data, topic_options

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.getenv('PORT', 8050)))
