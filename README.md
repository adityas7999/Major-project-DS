# The Data-Driven Social Engagement Initiative

Predicting and visualising YouTube engagement using NLP sentiment, day-one hype and title structure.

**Author:** <your name> | **Course:** Data Science Major Project

## Data source (please read)

This project uses a **public Kaggle dataset**, not data collected by me:

- **Dataset:** "Trending YouTube Video Statistics" (`datasnaek/youtube`) on Kaggle
- **Files used:** `USvideos.csv` (video statistics) and `UScomments.csv` (comments)
- **Link:** https://www.kaggle.com/datasets/datasnaek/youtube
- **Permission:** Use of Kaggle data was confirmed as allowed by the course support team.
- **License:** check the license shown on the Kaggle page before redistributing the raw files.

The data is downloaded automatically in Notebook 1 with `kagglehub`. Only videos that appeared on the YouTube trending page are included.

## Deliverables

| # | Deliverable | Where to find it |
|---|---|---|
| 1 | Fully functional Analytics Dashboard | `app.py` (Streamlit) |
| 2 | Structured Dataset | `cleaned_videos_v1.csv`, `cleaned_comments_v1.csv`, `master_dataset.csv` |
| 3 | NLP Sentiment Model (tags comments "Relatable" / "Neutral") | `01_Data_Prep_and_NLP.ipynb` (VADER, threshold compound >= 0.4) |
| 4 | Strategy Report | `Strategy_Report` file (see notes below) |
| 5 | The Content Series | <fill in: see notes below> |

## Project modules

1. **Notebook 1 - Data Prep and NLP:** cleaning, hype features, comment sentiment, builds `master_dataset.csv`
2. **Notebook 2 - Exploratory Data Analysis:** correlations, title length, categories, viral share
3. **Notebook 3 - Virality Prediction Engine:** Decision Tree and Random Forest, baseline comparison, cross-validation, permutation importance
4. **Dashboard:** filters, sentiment, hype and growth, title and category views, prediction model

## Key definitions

- **Engagement rate** = (likes + comments) / views, from the video's last day on the trending page
- **Viral** = top 20% by engagement rate (a relative label among trending videos)
- **Relatability ratio** = share of a video's comments scored strongly positive by VADER (needs at least 20 scored comments)
- **Hype velocity** = views on the video's first day on the trending page

## How to run

```bash
pip install -r requirements.txt
# Run notebooks 01 -> 02 -> 03 in order (creates master_dataset.csv)
streamlit run app.py
```

Keep `app.py` and `master_dataset.csv` in the same folder.

## Limitations

- Results show associations, not causes.
- Only trending videos are included, so findings may not apply to all of YouTube.
- Comment sentiment is collected after launch, so it describes audience reaction and cannot be set in advance.
- Only 20% of videos are labelled viral, so accuracy is reported next to the 80% "always not viral" baseline, together with AUC and recall.
