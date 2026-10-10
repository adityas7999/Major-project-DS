import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Page Configuration
st.set_page_config(page_title="Social Engagement Engine", layout="wide")
st.title("📈 Data-Driven Social Engagement Initiative")
st.markdown("Exploring and predicting YouTube engagement through NLP sentiment, day-one hype and title structure.")

CATEGORY_NAMES = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals", 17: "Sports",
    19: "Travel & Events", 20: "Gaming", 22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News & Politics", 26: "How-to & Style", 27: "Education", 28: "Science & Tech",
    29: "Nonprofits & Activism", 30: "Movies", 43: "Shows",
}
FEATURES = ['title_char_len', 'title_word_count', 'has_question_hook', 'has_exclamation_hook',
            'relatability_ratio', 'hype_velocity_per_day']
REQUIRED = ['video_id', 'title', 'category_id', 'views', 'engagement_rate',
            'relatability_ratio', 'hype_velocity_per_day']


def show(fig):
    """Render a plotly chart on both old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# 2. Data Ingestion (Cached for performance)
@st.cache_data
def load_data(path="master_dataset.csv"):
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"master_dataset.csv is missing columns: {missing}. Re-run Notebook 1.")

    for col in ['relatability_ratio', 'hype_velocity_per_day', 'engagement_rate', 'views']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['relatability_ratio', 'hype_velocity_per_day', 'engagement_rate']).copy()

    df['category_name'] = df['category_id'].map(CATEGORY_NAMES).fillna(df['category_id'].astype(str))
    df['category_label'] = df['category_id'].astype(str) + " - " + df['category_name']

    # Same title features as Notebook 3
    title = df['title'].astype(str)
    df['title_char_len'] = title.apply(len)
    df['title_word_count'] = title.apply(lambda x: len(x.split()))
    df['has_question_hook'] = title.apply(lambda x: int('?' in x))
    df['has_exclamation_hook'] = title.apply(lambda x: int('!' in x))
    df['log_hype_velocity'] = np.log1p(df['hype_velocity_per_day'])

    # "Viral" = top 20% engagement rate (threshold fixed on the FULL dataset so filters don't change the definition)
    threshold = df['engagement_rate'].quantile(0.80)
    df['is_viral_num'] = (df['engagement_rate'] >= threshold).astype(int)
    df['Status'] = np.where(df['is_viral_num'] == 1, 'Viral', 'Not viral')

    # Title-length quartiles built once, in the correct order
    q = pd.qcut(df['title_char_len'], 4, duplicates='drop')
    names = {iv: f"{int(np.floor(iv.left)) + 1}-{int(iv.right)} chars" for iv in q.cat.categories}
    ordered = list(names.values())
    df['title_len_group'] = pd.Categorical(q.map(names), categories=ordered, ordered=True)

    return df, threshold


@st.cache_resource
def train_model(_df, n_rows):
    """Same model as Notebook 3 (balanced Random Forest, 80/20 stratified split)."""
    x = pd.get_dummies(_df[FEATURES + ['category_id']], columns=['category_id'], drop_first=True)
    y = _df['is_viral_num']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample',
                                random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)

    pred = rf.predict(x_test)
    prob = rf.predict_proba(x_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, pred),
        'auc': roc_auc_score(y_test, prob),
        'recall': recall_score(y_test, pred),
        'precision': precision_score(y_test, pred, zero_division=0),
        'baseline': 1 - y_test.mean(),
    }

    perm = permutation_importance(rf, x_test, y_test, scoring='roc_auc', n_repeats=10,
                                  random_state=42, n_jobs=-1)
    imp = pd.DataFrame({'feature': x.columns, 'importance': perm.importances_mean})
    imp['feature'] = imp['feature'].where(~imp['feature'].str.startswith('category_id_'), 'category (all)')
    imp = imp.groupby('feature', as_index=False).sum().sort_values('importance', ascending=False)

    return rf, list(x.columns), metrics, imp


try:
    df, viral_threshold = load_data()
except FileNotFoundError:
    st.error("master_dataset.csv not found. Run Notebook 1 and place the file next to app.py.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

# 3. Sidebar Filtering Engine
st.sidebar.header("Filter Content Data")
category_options = sorted(df['category_label'].unique().tolist(), key=lambda s: int(s.split(' - ')[0]))
selected_categories = st.sidebar.multiselect(
    "YouTube categories", options=category_options, default=category_options,
    help="Remove categories to focus on specific niches."
)
min_group = st.sidebar.slider("Minimum videos per category (category chart)", 1, 100, 30)

filtered_df = df[df['category_label'].isin(selected_categories)] if selected_categories else df
if filtered_df.empty:
    st.warning("No videos match the current filters.")
    st.stop()

with st.expander("How to read this dashboard (definitions)"):
    st.markdown(f"""
- **Engagement rate** = (likes + comments) / views, taken from each video's last day on the trending page.
- **Viral** = top 20% by engagement rate (engagement rate ≥ {viral_threshold:.2%}). It means *highly engaging among trending videos*, not "viral across all of YouTube".
- **Relatability ratio** = share of a video's comments that VADER scores as strongly positive (compound ≥ 0.4).
- **Hype velocity** = views the video already had on its first day on the trending page (day-one momentum).
- All relationships shown are **associations**, not proof of cause.
""")

# 4. Top-Level Executive KPIs
st.header("Executive Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Videos Analyzed", f"{len(filtered_df):,}")
c2.metric("Viral Share", f"{filtered_df['is_viral_num'].mean():.1%}")
c3.metric("Median Engagement", f"{filtered_df['engagement_rate'].median():.2%}")
c4.metric("Avg Relatability", f"{filtered_df['relatability_ratio'].mean():.1%}")
c5.metric("Median Day-One Views", f"{int(filtered_df['hype_velocity_per_day'].median()):,}")

st.divider()

tab_sent, tab_hype, tab_title, tab_model, tab_data = st.tabs(
    ["💬 Sentiment", "🚀 Hype & Growth", "🔤 Title & Category", "🤖 Prediction Model", "🗂 Data"]
)

color_map = {'Viral': 'crimson', 'Not viral': 'gray'}

# --- Sentiment ---
with tab_sent:
    st.subheader("Emotion vs. Engagement")
    st.markdown("Do videos with more positive comment sections also have higher engagement?")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(
            filtered_df, x="relatability_ratio", y="engagement_rate", color="Status",
            hover_data=['title', 'views'], color_discrete_map=color_map, opacity=0.6,
            labels={"relatability_ratio": "Relatability Ratio", "engagement_rate": "Engagement Rate"}
        )
        # Cap Y-axis to 99th percentile to prevent outliers from squashing the chart
        fig.update_yaxes(range=[0, filtered_df['engagement_rate'].quantile(0.99)])
        show(fig)
    with col_b:
        fig = px.box(
            filtered_df, x="Status", y="relatability_ratio", color="Status",
            color_discrete_map=color_map, category_orders={"Status": ["Not viral", "Viral"]},
            labels={"relatability_ratio": "Relatability Ratio"}
        )
        fig.update_layout(showlegend=False)
        show(fig)
    rho = filtered_df[['relatability_ratio', 'engagement_rate']].corr(method='spearman').iloc[0, 1]
    st.caption(f"Spearman correlation between relatability and engagement for the selected videos: {rho:+.2f}. "
               "Positive comments and likes both come from an already-happy audience, so this is a link, not a lever.")

# --- Hype & growth ---
with tab_hype:
    st.subheader("Day-One Hype and Growth")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("How many views do videos have on their first day trending?")
        fig = px.histogram(filtered_df, x="log_hype_velocity", nbins=40, color_discrete_sequence=['purple'],
                           labels={"log_hype_velocity": "Log(Day-one views + 1)"})
        show(fig)
    with col_b:
        if 'final_views' in filtered_df.columns:
            st.markdown("Day-one views vs. total views by the last trending day.")
            fig = px.scatter(
                filtered_df, x="hype_velocity_per_day", y="final_views", color="Status",
                color_discrete_map=color_map, opacity=0.5, log_x=True, log_y=True,
                hover_data=['title'],
                labels={"hype_velocity_per_day": "Day-one views", "final_views": "Final views"}
            )
            show(fig)
        else:
            st.info("final_views column not found - re-run Notebook 1 to see this chart.")

    if 'days_trending' in filtered_df.columns:
        st.markdown("Does staying on the trending page longer go with higher engagement?")
        grp = filtered_df.assign(days=filtered_df['days_trending'].clip(upper=5).astype(int).astype(str).replace({'5': '5+'}))
        days_stats = grp.groupby('days').agg(viral_share=('is_viral_num', 'mean'),
                                             videos=('video_id', 'count')).reset_index()
        fig = px.bar(days_stats, x='days', y='viral_share', text='videos',
                     category_orders={'days': ['1', '2', '3', '4', '5+']},
                     labels={'days': 'Days on trending page', 'viral_share': 'Share of videos that are viral'})
        fig.update_traces(texttemplate='n=%{text}')
        fig.update_yaxes(tickformat='.0%')
        show(fig)
        st.caption("Days trending is measured over the same period as engagement, so it describes what happened after launch; it is not a predictor.")

# --- Title & category ---
with tab_title:
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Title Architecture")
        st.markdown("Share of videos that are viral, by title-length quartile.")
        title_stats = filtered_df.groupby('title_len_group', observed=True)['is_viral_num'].mean().reset_index()
        title_stats['title_len_group'] = title_stats['title_len_group'].astype(str)
        fig = px.bar(title_stats, x='title_len_group', y='is_viral_num', color_discrete_sequence=['royalblue'],
                     labels={'title_len_group': 'Title length', 'is_viral_num': 'Share viral'})
        fig.add_hline(y=filtered_df['is_viral_num'].mean(), line_dash="dash", line_color="gray",
                      annotation_text="Overall viral share")
        fig.update_yaxes(tickformat='.0%')
        show(fig)
    with col_d:
        st.subheader("Category Performance")
        st.markdown(f"Median engagement rate by niche (categories with at least {min_group} videos).")
        cat_stats = filtered_df.groupby(['category_id', 'category_name']).agg(
            median_engagement=('engagement_rate', 'median'), videos=('video_id', 'count')).reset_index()
        cat_stats = cat_stats[cat_stats['videos'] >= min_group].sort_values('median_engagement', ascending=False).head(10)
        if cat_stats.empty:
            st.info("No category has enough videos. Lower the minimum in the sidebar.")
        else:
            fig = px.bar(cat_stats, x='category_name', y='median_engagement', text_auto='.2%',
                         hover_data=['videos'], color='median_engagement', color_continuous_scale='viridis',
                         labels={"category_name": "Category", "median_engagement": "Median Engagement Rate"})
            show(fig)

# --- Prediction model ---
with tab_model:
    st.subheader("Virality Prediction Engine")
    rf, model_cols, metrics, importance = train_model(df, len(df))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}", help="Compare with the baseline below")
    m2.metric("Baseline (always 'not viral')", f"{metrics['baseline']:.1%}")
    m3.metric("AUC", f"{metrics['auc']:.2f}")
    m4.metric("Viral recall", f"{metrics['recall']:.0%}", help="Share of truly viral videos the model catches")
    st.caption("Balanced Random Forest on a held-out 20% test set. Only 20% of videos are viral, so AUC and recall matter more than accuracy.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**What the model relies on** (drop in AUC when the feature is shuffled)")
        fig = px.bar(importance.head(8).iloc[::-1], x='importance', y='feature', orientation='h',
                     color_discrete_sequence=['teal'], labels={'importance': 'Permutation importance', 'feature': ''})
        show(fig)
    with col_b:
        st.markdown("**Try a video idea**")
        in_title = st.text_input("Title", "I tried the world's spiciest noodles!")
        in_cat = st.selectbox("Category", category_options, index=0)
        in_relat = st.slider("Expected share of positive comments", 0.0, 1.0, float(df['relatability_ratio'].median()), 0.01)
        in_hype = st.number_input("Day-one views", min_value=0, value=int(df['hype_velocity_per_day'].median()), step=1000)

        row = pd.DataFrame([{
            'title_char_len': len(in_title), 'title_word_count': len(in_title.split()),
            'has_question_hook': int('?' in in_title), 'has_exclamation_hook': int('!' in in_title),
            'relatability_ratio': in_relat, 'hype_velocity_per_day': in_hype,
            'category_id': int(in_cat.split(' - ')[0]),
        }])
        row = pd.get_dummies(row, columns=['category_id']).reindex(columns=model_cols, fill_value=0)
        prob = rf.predict_proba(row)[0, 1]
        st.metric("Estimated chance of being in the top 20% engagement", f"{prob:.0%}")
        st.caption("Typical videos score about 20%. This is a rough ranking score from a moderate model, not a guarantee. "
                   "Positive comments and day-one views are only known after launch.")

# --- Data ---
with tab_data:
    st.subheader("Raw Analytical Matrix")
    show_cols = [c for c in ['video_id', 'title', 'category_name', 'views', 'relatability_ratio',
                             'hype_velocity_per_day', 'engagement_rate', 'Status'] if c in filtered_df.columns]
    st.dataframe(filtered_df[show_cols].head(200))
    st.download_button("Download filtered data (CSV)", filtered_df.to_csv(index=False).encode('utf-8'),
                       file_name="filtered_videos.csv", mime="text/csv")
