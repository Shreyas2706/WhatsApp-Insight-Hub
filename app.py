import pandas as pd
import streamlit as st
import preproccesor, helper
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI Enhancements
st.set_page_config(page_title='WhatsApp Insight Hub', layout='wide')
st.markdown('<style>body {background-color: #f7f9fc;}</style>', unsafe_allow_html=True)
st.title("💬 WhatsApp Insight Hub")
st.markdown("### Designed and Developed by Shreyas Singh - Shevy")
st.markdown("### Analyze your WhatsApp chat data with powerful insights and beautiful visualizations!")

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
        .stButton button {
            width: 100%;
        }
        .stDataFrame {
            overflow-x: auto;
        }
        .stTabs [role="tablist"] {
            flex-wrap: wrap;
        }
        .stTabs [role="tab"] {
            flex: 1 1 auto;
            text-align: center;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 WhatsApp Chat Analyzer")
st.sidebar.markdown("Upload your chat file and dive into detailed analysis.")
uploaded_file = st.sidebar.file_uploader("📂 Upload your WhatsApp chat file (.txt)")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preproccesor.preprocess(data)
    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("🔍 Select User for Analysis", user_list)

    if st.sidebar.button("Analyze"):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        # Summary Statistics
        st.markdown("## 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media_messages)
        col4.metric("Links Shared", num_links)

        # Timeline Tabs
        st.markdown("## 📅 Timelines")
        tab1, tab2 = st.tabs(["Monthly Timeline", "Daily Timeline"])

        with tab1:
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(timeline['time'], timeline['message'], marker='o', linestyle='-', color='green', linewidth=2, markersize=6)
            ax.set_title("Monthly Message Timeline", fontsize=16)
            ax.set_xlabel("Month-Year")
            ax.set_ylabel("Number of Messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with tab2:
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            ax.set_title("Daily Message Timeline", fontsize=16)
            ax.set_xlabel("Timeline")
            ax.set_ylabel("Number Of Messages")
            st.pyplot(fig)

        # Activity Map
        st.markdown("## 🗺️ Activity Map")
        busy_day = helper.week_activity_map(selected_user, df)
        fig, ax = plt.subplots()
        sns.barplot(x=busy_day.index, y=busy_day.values, ax=ax, palette='viridis')
        ax.set_title("Most Active Days")
        ax.set_xlabel("Day", color='red')
        ax.set_ylabel("Number Of Messages", color='blue')
        st.pyplot(fig)

        # Emoji Analysis
        st.markdown("## 😀 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        col1.dataframe(emoji_df)
        fig, ax = plt.subplots()
        ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f%%")
        col2.pyplot(fig)

        # WordCloud
        st.markdown("## ☁️ WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Top Words
        st.markdown("## 📚 Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        sns.barplot(y=most_common_df.iloc[:, 0], x=most_common_df.iloc[:, 1], ax=ax, palette='coolwarm')
        ax.set_title("Top Words Usage")
        st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Sentiment Analysis
        st.markdown("## 😊 Sentiment Analysis")
        sentiment = helper.sentiment_analysis(selected_user, df)
        fig, ax = plt.subplots()
        sns.barplot(x=list(sentiment.keys()), y=list(sentiment.values()), palette='pastel', ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # Footer
        st.markdown("---")
        st.markdown("Designed and Developed by Shreyas Singh - Shrevi")

else:
    st.sidebar.info("Upload a WhatsApp chat file to start analysis.")

