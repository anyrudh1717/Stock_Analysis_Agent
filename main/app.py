from flask import Flask, render_template, request, redirect, url_for, session, flash
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import plotly.graph_objects as go
from crewai import Agent, Task, Crew, Process, LLM
import requests
import http.client
import config
import json
from bs4 import BeautifulSoup
from textblob import TextBlob
import os

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Initialize LLM
llm = LLM(
    model="groq/llama3-8b-8192",
    api_key=config.GROQ_API_KEY
)

# Alpha Vantage API Key
API_KEY = config.ALPHA_VANTAGE_API_KEY
CSV_FILE_PATH = os.path.join('static', 'stock_symbols.csv')

# Login credentials
USER_CREDENTIALS = {
    'admin': 'password123'
}

# Helper functions
def get_stock_symbols():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        return df['symbol'].tolist()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def get_intraday_data(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval='5min', outputsize='compact')
    return data

def get_latest_price(df):
    return df.iloc[0]['4. close'] if not df.empty else None

def get_stock_news(symbol):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": f"{symbol} stock news",
        "num": 10,
        "tbs": "qdr:d"
    })
    headers = {
        'X-API-KEY': config.SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/news", payload, headers)
    res = conn.getresponse()
    data = res.read()
    response_data = json.loads(data.decode("utf-8"))
    return [article.get('link', '') for article in response_data.get('news', [])]

def fetch_article_content(url):
    try:
        res = requests.get(url)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            return ' '.join([p.get_text() for p in soup.find_all('p')])
        return "Failed to fetch content"
    except Exception as e:
        return str(e)

def analyze_article_content(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive sentiment detected"
    elif polarity < 0:
        return "Negative sentiment detected"
    return "Neutral sentiment detected"

def get_article_details(links):
    results = []
    for link in links:
        content = fetch_article_content(link)
        sentiment = analyze_article_content(content)
        results.append(f"Article from {link}: {sentiment}")
    return results

# Create agents using keyword arguments
def create_agents():
    classifier = Agent(
        role="stock classifier",
        goal="Accurately classify the stock based on its performance and market news",
        allow_delegation=False,
        backstory="Classify stocks as Bullish, Bearish, or Neutral based on market data and news.",
        llm=llm
    )
    recommender = Agent(
        role="stock recommender",
        goal="Provide a buy, sell, or hold recommendation based on stock classification.",
        allow_delegation=False,
        backstory="Provide a clear recommendation. If sentiment is negative, suggest 'Sell'.",
        llm=llm
    )
    researcher = Agent(
        role="stock news researcher",
        goal="Research and analyze stock news to provide insights.",
        allow_delegation=False,
        backstory="Analyze news and extract insights in 10 detailed points without sharing links.",
        llm=llm
    )
    return classifier, recommender, researcher

# Login route
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        if USER_CREDENTIALS.get(user) == pwd:
            session['username'] = user
            return redirect(url_for('home'))
        flash('Invalid credentials.')
    return render_template('login.html')

# Main route
@app.route('/home', methods=['GET', 'POST'])
def home():
    stock_symbols = get_stock_symbols()
    output_text = ""
    graph_html = None
    stock_price = None
    price_change_text = None
    price_change_class = None

    if request.method == 'POST':
        symbol = request.form['stock_symbol']
        df = get_intraday_data(symbol)
        news_links = get_stock_news(symbol)
        article_details = get_article_details(news_links)

        if df.empty:
            output_text = f"No data available for symbol {symbol}"
        else:
            stock_price = get_latest_price(df)
            old_price = df.iloc[-1]['4. close']
            price_change = ((stock_price - old_price) / old_price) * 100
            price_change_text = f"{price_change:+.2f}%"
            price_change_class = "positive" if price_change >= 0 else "negative"

            stock_data = {
                "symbol": symbol,
                "price": stock_price,
                "news": article_details
            }

            classifier, recommender, researcher = create_agents()

            classify_stock = Task(
                description=f"Classify the stock based on the data: {stock_data}",
                agent=classifier,
                expected_output="Bullish, Bearish, or Neutral"
            )
            recommend_stock = Task(
                description=f"Provide a recommendation for the stock: {stock_data}",
                agent=recommender,
                expected_output="Buy, Sell, or Hold with explanation"
            )
            research_news = Task(
                description=f"Research and analyze these news articles: {article_details}",
                agent=researcher,
                expected_output="Insights from the news articles"
            )

            crew = Crew(
                agents=[classifier, recommender, researcher],
                tasks=[classify_stock, recommend_stock, research_news],
                process=Process.sequential
            )

            try:
                output_text = crew.kickoff()
            except Exception as e:
                output_text = f"Error during execution: {e}"

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['4. close'],
                mode='lines',
                name="Closing Price",
                line=dict(color='red')
            ))
            fig.update_layout(
                title=f"{symbol} Intraday Stock Data (5-min Interval)",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                template="seaborn",
                xaxis_rangeslider_visible=True
            )
            graph_html = fig.to_html(full_html=False)

    return render_template(
        'index.html',
        output_text=output_text,
        graph_html=graph_html,
        stock_symbols=stock_symbols,
        stock_price=stock_price,
        price_change_text=price_change_text,
        price_change_class=price_change_class
    )

if __name__ == '__main__':
    app.run(debug=True)
