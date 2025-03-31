from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob

# Initialize URL extractor
extract = URLExtract()

# Helper function to filter data by user
def filter_by_user(selected_user, df):
    if selected_user != 'Overall':
        return df[df['user'] == selected_user]
    return df

# Function to fetch basic statistics
def fetch_stats(selected_user, df):
    df = filter_by_user(selected_user, df)

    # Number of messages
    num_messages = df.shape[0]

    # Total number of words
    words = [word for message in df['message'] for word in message.split()]

    # Number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # Number of links shared
    links = [url for message in df['message'] for url in extract.find_urls(message)]

    return num_messages, len(words), num_media_messages, len(links)

# Advanced Statistics - Active Hours
def active_hours(df):
    active_hours = df['hour'].value_counts().head(10)
    return active_hours

# Advanced Statistics - Active Days
def active_days(df):
    active_days = df['day_name'].value_counts().head(7)
    return active_days

# Message Length Analysis
def message_length_analysis(selected_user, df):
    df = filter_by_user(selected_user, df)
    df['message_length'] = df['message'].apply(len)
    avg_length = df['message_length'].mean()
    longest_message = df.loc[df['message_length'].idxmax()]['message']
    shortest_message = df.loc[df['message_length'].idxmin()]['message']
    return avg_length, longest_message, shortest_message

# Sentiment Analysis using TextBlob
def sentiment_analysis(selected_user, df):
    df = filter_by_user(selected_user, df)
    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    def analyze_sentiment(message):
        polarity = TextBlob(message).sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    df['Sentiment'] = df['message'].apply(analyze_sentiment)
    sentiments_count = df['Sentiment'].value_counts().to_dict()

    # Update the sentiment counts
    for key in sentiments_count:
        sentiments[key] = sentiments_count[key]

    return sentiments

# Most Busy Users
def most_busy_users(df):
    user_counts = df['user'].value_counts().head()
    user_percent = (df['user'].value_counts() / df.shape[0] * 100).round(2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return user_counts, user_percent

# Word Cloud
def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())

    df = filter_by_user(selected_user, df)
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    def clean_message(message):
        return " ".join(word for word in message.lower().split() if word not in stop_words)

    temp['message'] = temp['message'].apply(clean_message)
    text = " ".join(temp['message'])

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(text)

# Most Common Words
def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())

    df = filter_by_user(selected_user, df)
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    words = [word for message in temp['message'] for word in message.lower().split() if word not in stop_words]
    most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=['Word', 'Frequency'])
    return most_common_df

# Emoji Analysis
def emoji_helper(selected_user, df):
    df = filter_by_user(selected_user, df)
    emojis = [c for message in df['message'] for c in message if c in emoji.EMOJI_DATA]
    emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=['Emoji', 'Count'])
    return emoji_df

# Monthly Timeline
def monthly_timeline(selected_user, df):
    df = filter_by_user(selected_user, df)
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline

# Daily Timeline
def daily_timeline(selected_user, df):
    df = filter_by_user(selected_user, df)
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

# Weekly Activity Map
def week_activity_map(selected_user, df):
    df = filter_by_user(selected_user, df)
    return df['day_name'].value_counts()

# Monthly Activity Map
def month_activity_map(selected_user, df):
    df = filter_by_user(selected_user, df)
    return df['month'].value_counts()

# Activity heatmap
def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # Check if 'time' column exists
    if 'time' not in df.columns:
        # If not, try to extract time from 'date' column
        df['time'] = pd.to_datetime(df['date']).dt.time
    # Extracting day name and period
    df['day_name'] = pd.to_datetime(df['date']).dt.day_name()
    df['hour'] = pd.to_datetime(df['time'].astype(str)).dt.hour
    df['period'] = df['hour'].apply(lambda x: f'{x:02d}:00-{(x+1)%24:02d}:00')
    heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return heatmap








