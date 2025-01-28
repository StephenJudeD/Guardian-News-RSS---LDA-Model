Guardian News Topic Explorer
============================

This repository contains a Dash app that fetches articles from the Guardian (via their API), processes them using Natural Language Processing (LDA topic modeling, t-SNE visualization, etc.), and displays interactive plots in a dashboard. It's perfect for anyone wanting to explore the topics and clusters of recent news articles without digging through them one by one.

Features
--------

1.  **Article Fetching**\
    Retrieves Guardian articles in a user-defined date range (e.g., last day, last week, last month).
2.  **Topic Modeling (LDA)**\
    Discovers hidden topics in the news articles using Gensim's LDA model.
3.  **NLP Preprocessing**\
    Tokenization, removing stopwords (including custom stopwords), detecting bigrams/trigrams, etc.
4.  **Visualization**
    -   Topic word distributions (bar charts)
    -   Word clouds for top terms in a topic
    -   3D t-SNE plot of articles in topic space
    -   Bubble chart of document lengths over time
    -   Bar chart of the most common bigrams/trigrams
5.  **Article Table**\
    Displays detailed info (title, date published, topic probabilities).

Requirements
------------

To run this app, you'll need:

-   Python 3.8+
-   A Guardian API key. You can request one at:\
    <https://open-platform.theguardian.com/access/>
-   The necessary Python libraries, mostly installed via `requirements.txt`:
    -   Dash (with Plotly, dash-table, etc.)
    -   dash-bootstrap-components
    -   Gensim
    -   NLTK
    -   Scikit-learn
    -   WordCloud
    -   Pandas, NumPy
    -   python-dotenv
    -   Others as needed

Installation
------------

1.  Clone this repository:

    bash

    Copy

    ```
    git clone https://github.com/<YOUR_USERNAME>/guardian-news-topic-explorer.git
    cd guardian-news-topic-explorer

    ```

2.  (Optional) Create and activate a virtual environment:

    bash

    Copy

    ```
    python3 -m venv venv
    source venv/bin/activate

    ```

3.  Install dependencies using `pip`:

    bash

    Copy

    ```
    pip install -r requirements.txt

    ```

4.  Download NLTK data (if you haven't already):

    python

    RunCopy

    ```
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    ```

5.  Create a local `.env` file to store environment variables (see below).

Environment Variables
---------------------

1.  **Guardian API Key**\
    In order to fetch articles from the Guardian, you need a Guardian API key. Put it in a `.env` file in the root directory of this project:

    Copy

    ```
    GUARDIAN_API_KEY=YOUR_GUARDIAN_API_KEY_HERE

    ```

2.  **Heroku / Docker**\
    For cloud deployments (e.g., Heroku), set the `GUARDIAN_API_KEY` in the config or environment settings.

Local Usage
-----------

1.  Make sure your `.env` file is properly set with your Guardian API key.
2.  Run the app locally:

    bash

    Copy

    ```
    python app.py

    ```

3.  Open your browser to the displayed Dash URL (typically <http://127.0.0.1:8050/>) to explore the dashboard.

Heroku Deployment
-----------------

1.  **Prerequisites**

    -   A Heroku account and the Heroku CLI installed locally.
    -   This repository connected to Heroku, or you can create a new Heroku app and push your code there.
2.  **Set Environment Variables**\
    In your Heroku dashboard or via CLI, set the Guardian API key:

    bash

    Copy

    ```
    heroku config:set GUARDIAN_API_KEY=YOUR_GUARDIAN_API_KEY_HERE

    ```

    Heroku will now expose `GUARDIAN_API_KEY` to the Python environment at runtime.

3.  **Procfile**\
    Check that you have a `Procfile` in your repo specifying how to run the app. For example:

    Copy

    ```
    web: gunicorn app:server

    ```

    This tells Heroku to run the Dash app using Gunicorn.

4.  **Push to Heroku**

    bash

    Copy

    ```
    git add .
    git commit -m "Initial commit"
    heroku git:remote -a YOUR_HEROKU_APP_NAME
    git push heroku main

    ```

5.  **Confirm Deployment**\
    Once the build is finished, navigate to <https://YOUR_HEROKU_APP_NAME.herokuapp.com>. You should see the Guardian News Topic Explorer running.

Tips to Avoid Timeouts
----------------------

-   Heroku will terminate any request taking longer than 30 seconds. If you're fetching large date ranges or dealing with heavy LDA computations, consider:
    1.  Reducing the number of passes or chunk sizes in LDA.
    2.  Using server-side caching so you don't re-train the model every time.
    3.  Offloading long-running tasks to a worker dyno (Celery/RQ).

Contributing
------------

Pull requests and issues are welcome! If you have ideas, bug reports, or improvements, feel free to open an issue or submit a PR.

License
-------

MIT License, or something like that!
