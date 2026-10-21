
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data.csv")

if "review" not in df.columns:
    raise ValueError("CSV file must contain a column named 'review'.")

df["review"] = df["review"].fillna("").astype(str)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Classify sentiment
def classify_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Detect basic emotion using word lexicons
emotion_lexicon = {
    "Happy": [
        "happy", "love", "wonderful", "excellent",
        "delighted", "amazing", "great", "best", "joy"
    ],
    "Anger": [
        "hate", "angry", "furious", "horrible",
        "annoyed", "worst", "terrible"
    ],
    "Sadness": [
        "sad", "disappointed", "unhappy", "regret",
        "upset", "sorry"
    ],
    "Fear": [
        "fear", "scared", "afraid", "worried",
        "anxious", "risk"
    ]
}


def detect_emotion(text):
    words = text.lower().split()
    counts = Counter()

    for emotion, keywords in emotion_lexicon.items():
        for keyword in keywords:
            counts[emotion] += sum(
                1 for word in words
                if word.strip(".,!?;:") == keyword
            )

    if not counts or max(counts.values()) == 0:
        return "No clear emotion"

    return counts.most_common(1)[0][0]


# Perform analysis
df["sentiment"] = df["review"].apply(classify_sentiment)
df["emotion"] = df["review"].apply(detect_emotion)

# Save analyzed data
df.to_csv("outputs/analyzed_reviews.csv", index=False)

# Display results
print("\n========== SENTIMENT ANALYSIS ==========")
print(df.to_string(index=False))

print("\n========== SENTIMENT SUMMARY ==========")
print(df["sentiment"].value_counts())

print("\n========== EMOTION SUMMARY ==========")
print(df["emotion"].value_counts())


# Chart 1: Sentiment distribution
sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values
)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/sentiment_distribution.png")
plt.close()


# Chart 2: Sentiment pie chart
plt.figure(figsize=(7, 7))
sentiment_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Sentiment Percentage")
plt.ylabel("")
plt.tight_layout()
plt.savefig("outputs/sentiment_pie_chart.png")
plt.close()


# Chart 3: Emotion distribution
emotion_counts = df["emotion"].value_counts()

plt.figure(figsize=(9, 5))
sns.barplot(
    x=emotion_counts.index,
    y=emotion_counts.values
)
plt.title("Emotion Distribution")
plt.xlabel("Detected Emotion")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/emotion_distribution.png")
plt.close()


# Chart 4: Word cloud
all_reviews = " ".join(df["review"].tolist())

if all_reviews.strip():
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(all_reviews)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Customer Review Word Cloud")
    plt.tight_layout()
    plt.savefig("outputs/wordcloud.png")
    plt.close()


print("\nAnalysis completed successfully!")
print("Analyzed data: outputs/analyzed_reviews.csv")
print("Charts saved inside the outputs folder.")
