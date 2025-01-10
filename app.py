# app.py
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from guardian_fetcher import GuardianFetcher
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
from dotenv import load_dotenv
from functools import lru_cache

@lru_cache(maxsize=32)
def process_articles(articles_df):
    # Your processing code here
    return processed_data
    
# Load local .env file if it exists (development)
load_dotenv()

# Get API key from environment (works with Heroku config vars)
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
if not GUARDIAN_API_KEY:
    raise ValueError("⚠️ No GUARDIAN_API_KEY found in environment!")

# For Heroku's port binding
port = int(os.getenv('PORT', 8050))

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Guardian fetcher
guardian = GuardianFetcher(GUARDIAN_API_KEY)

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Guardian News Topic Explorer 📰", className="text-center mb-4"),
            html.P("Interactive analysis of Guardian articles using LDA topic modeling", 
                  className="text-center text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Controls", className="card-title"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=(datetime.now() - timedelta(days=30)).date(),
                        end_date=datetime.now().date(),
                        className="mb-3"
                    ),
                    dcc.Slider(
                        id='topic-slider',
                        min=2, max=10,
                        value=5,
                        marks={i: str(i) for i in range(2, 11)},
                        className="mb-3"
                    ),
                    dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=html.Div(id="loading-output")
                    )
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='topic-explorer')
        ])
    ])
])

@app.callback(
    Output('topic-explorer', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('topic-slider', 'value')]
)
def update_topics(start_date, end_date, num_topics):
    # Fetch articles
    df = guardian.fetch_articles(days_back=30)
    
    # Preprocessing
    stop_words = set(stopwords.words('english'))
    texts = df['content'].apply(lambda x: [
        word.lower() for word in word_tokenize(str(x))
        if word.isalnum() and word.lower() not in stop_words
    ])
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10
    )
    
    # Create visualization
    fig = create_interactive_topic_explorer(lda_model, corpus, dictionary, df)
    
    return fig

def create_interactive_topic_explorer(lda_model, corpus, dictionary, df, num_words=10):
    # Get document-topic distributions
    doc_topics = []
    for doc in corpus:
        topic_weights = [0] * lda_model.num_topics
        for topic, weight in lda_model[doc]:
            topic_weights[topic] = weight
        doc_topics.append(topic_weights)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{"type": "table"}, {"type": "table"}]],
        subplot_titles=("Topic Word Distributions", "Article Details"),
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    
    # Add topic bars
    for topic_idx in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_idx, num_words)
        words, probs = zip(*topic_words)
        
        fig.add_trace(
            go.Bar(
                x=probs,
                y=words,
                orientation='h',
                name=f'Topic {topic_idx + 1}',
                marker_color=colors[topic_idx % len(colors)],
                customdata=[[topic_idx, w] for w in words],
                hovertemplate=(
                    "<b>Topic %{customdata[0]}</b><br>" +
                    "Word: %{y}<br>" +
                    "Probability: %{x:.3f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )
    
    # Add article details
    articles_data = pd.DataFrame({
        'Title': df['title'],
        'Section': df['section'],
        'Published': df['published'].dt.strftime('%Y-%m-%d %H:%M'),
        'Topic Distribution': [
            '<br>'.join([f'Topic {i+1}: {w:.3f}' 
                        for i, w in enumerate(weights)])
            for weights in doc_topics
        ]
    })
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(articles_data.columns),
                fill_color='rgba(50, 50, 50, 0.8)',
                align=['left'] * 4,
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[articles_data[col] for col in articles_data.columns],
                fill_color='rgba(30, 30, 30, 0.8)',
                align=['left'] * 4,
                font=dict(color='white', size=11),
                height=30
            )
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        template='plotly_dark',
        showlegend=True,
        title={
            'text': 'Interactive Topic Explorer',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        }
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
