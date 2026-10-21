
# CodeAlpha Sentiment Analysis

## Overview
This project analyzes customer reviews using Natural Language Processing (NLP).

## Features
- Positive, Negative and Neutral sentiment classification
- Basic emotion detection using a keyword lexicon
- Sentiment distribution bar chart
- Sentiment percentage pie chart
- Emotion distribution chart
- Customer review word cloud
- Export of analyzed reviews to CSV

## Technologies
- Python
- Pandas
- Matplotlib
- Seaborn
- VADER Sentiment
- WordCloud

## How to Run
1. Install Python.
2. Install dependencies using `pip install -r requirements.txt`.
3. Place the review dataset in `data.csv`.
4. Run `python main.py`.

## Output
The program saves analyzed reviews and visualizations inside the outputs folder.

## Note
The included dataset contains sample reviews for demonstration.
Emotion detection uses a basic keyword lexicon and may not identify context or sarcasm correctly.
