import streamlit as st
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
from openai import OpenAI
from collections import defaultdict
import tiktoken
import random
import textwrap
from dotenv import load_dotenv
import os
load_dotenv()

df = pd.DataFrame()
# import data
for file in os.listdir('data'):
    if file.endswith('.csv'):
        loop_df = pd.read_csv(f'data/{file}')
        df = pd.concat([df, loop_df])

print(f'Total rows in full dataframe: {df.shape[0]}')
# only allow data from 2025-01-01 forward
df = df[df['originalTimestamp'] >= '2025-01-01']



# sidebar menu
min_date = datetime.strptime(df['originalTimestamp'].min().split()[0], '%Y-%m-%d')
max_date = datetime.strptime(df['originalTimestamp'].max().split()[0], '%Y-%m-%d')

date_range = st.sidebar.slider(
    "Date Range Selector",
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

slider_start = date_range[0].strftime('%Y-%m-%d')
slider_end = date_range[1].strftime('%Y-%m-%d')

df = df[df['originalTimestamp'] >= slider_start]
df = df[df['originalTimestamp'] <= slider_end]


num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=7)




# data processing
query = """
with base as (
    SELECT DATE(originalTimestamp) as the_date, COALESCE(userId, anonymousId) as user_id
    FROM df 
)

select the_date, count(distinct user_id) as dau 
from base
where the_date between '{}' and '{}'
group by the_date
order by the_date
""".format(slider_start, slider_end)
dau = duckdb.sql(query).df()

query = """
with base as (
    SELECT DATE_TRUNC('month', DATE(originalTimestamp)) as month_year, 
           COALESCE(userId, anonymousId) as user_id
    FROM df 
)

select month_year, count(distinct user_id) as monthly_active_users
from base
group by month_year
order by month_year
"""
mau = duckdb.sql(query).df()


query = """
SELECT DATE(originalTimestamp) as the_date, count(distinct messageId) as page_views
FROM df 
WHERE event_type = 'page'
GROUP BY DATE(originalTimestamp)
"""
pvs = duckdb.sql(query).df()

query = """
with base as (
    SELECT DATE(originalTimestamp) as the_date, 
           COALESCE(userId, anonymousId) as user_id,
           COUNT(DISTINCT messageId) as user_page_views
    FROM df 
    WHERE event_type = 'page'
    GROUP BY DATE(originalTimestamp), COALESCE(userId, anonymousId)
)

SELECT the_date, AVG(user_page_views) as avg_page_views_per_user
FROM base
GROUP BY the_date
ORDER BY the_date
"""
page_views_per_user = duckdb.sql(query).df()

query = """
SELECT "page.referrer" as referrer, count(*) as count
FROM df 
WHERE "page.referrer" is not null
AND event_type = 'page'
GROUP BY "page.referrer"
ORDER BY count DESC
"""
referrers = duckdb.sql(query).df()

query = """
SELECT 
    "userAgentData.platform" as platform,
    count(*) as count
FROM df 
WHERE "userAgentData.platform" is not null
AND event_type = 'page'
GROUP BY "userAgentData.platform"
ORDER BY count DESC
"""
platform_mobile = duckdb.sql(query).df()

query = """
with impressions as (
    SELECT date(originalTimestamp) as the_date, 
    COALESCE(userId, anonymousId) as user_id, 
    unnest(string_split(business_id, ', ')) as single_business_id
    FROM df
    WHERE event_type = 'impression'
    and sponsored_listing = 'False'
)

, imp_agg as (
select the_date, user_id, count(distinct single_business_id) as business_count
from impressions
group by the_date, user_id
order by the_date, user_id
)

select the_date, avg(business_count) as avg_business_count
from imp_agg
group by the_date
order by the_date
"""
search_results = duckdb.sql(query).df()

query = """
with impressions as (
    SELECT date(originalTimestamp) as the_date, 
    COALESCE(userId, anonymousId) as user_id, 
    sponsored_listing,
    unnest(string_split(business_id, ', ')) as single_business_id
    FROM df
    WHERE event_type = 'impression'
)

, imp_agg as (
select the_date, single_business_id, sponsored_listing, count(distinct user_id) as num_impressions
from impressions
group by the_date, single_business_id, sponsored_listing
order by the_date, single_business_id, sponsored_listing
)

select the_date, sponsored_listing, avg(num_impressions) as avg_impressions
from imp_agg
group by the_date, sponsored_listing
order by the_date, sponsored_listing
"""
impressions = duckdb.sql(query).df()

query = """
with t1 as (
    SELECT date(originalTimestamp) as the_date, business_id, count(distinct messageId) as page_views
    FROM df 
    WHERE event_type = 'page'
    and business_id is not null
    group by date(originalTimestamp), business_id
)

select the_date, avg(page_views) as avg_page_views
from t1
group by the_date
order by the_date
"""
page_views_per_business = duckdb.sql(query).df()

query = """
with t1 as (
    SELECT date(originalTimestamp) as the_date, business_id, count(distinct messageId) as button_clicks
    FROM df 
    WHERE event_type = 'button_click'
    and button_name = 'visit_website'
    group by date(originalTimestamp), business_id
)

select the_date, avg(button_clicks) as avg_button_clicks
from t1
group by the_date
order by the_date
"""
button_clicks = duckdb.sql(query).df()


query = """
select DATE_TRUNC('month', DATE(originalTimestamp)) as month_year, count(distinct userId) as signups
from df
where event_type = 'button_click'
and button_name = 'faith'
group by DATE_TRUNC('month', DATE(originalTimestamp))
order by month_year
"""
signups = duckdb.sql(query).df()





query = """
SELECT lower(search_text) as search_text
FROM df
WHERE event_type = 'search'
and search_text is not null
"""
search_counts = duckdb.sql(query).df()

TEXT_COL = 'search_text'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(
    search_counts[TEXT_COL].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

km = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
km_labels = km.fit_predict(embeddings)
search_counts['kmeans_label']   = km_labels

kmeans_counts = search_counts.groupby(['kmeans_label', 'search_text']).size().reset_index(name='count')
kmeans_counts = kmeans_counts.sort_values(['kmeans_label', 'count'], ascending=[True, False])

MODEL = "gpt-3.5-turbo"
client = OpenAI()

# Helper: ensure prompt stays in model’s token limit
enc = tiktoken.encoding_for_model(MODEL)
def too_long(rows, max_tokens=4000):
    prompt = ' '.join(rows)
    return (len(enc.encode(prompt)) + 30) > max_tokens

def get_cluster_name(rows):
    if too_long(rows):
        prompt_rows = rows[:20] + rows[-20:]
    else:
        prompt_rows = rows

    user_prompt = f"""Given this data {str(prompt_rows)},
    what is a good one or two word category for a search term scatter plot to understand what users on a business directory are searching for.
    Only return the category name."""
    
    response = client.responses.create(
        model=MODEL,
        instructions="You are a coding assistant that only gives one or two word answers.",
        input=user_prompt
    )

    return response.output_text

# Get AI-generated names for each cluster
kmeans_ai_names = {}

# Get KMeans cluster names
for label in kmeans_counts['kmeans_label'].unique():
    if label == -1:  # Skip noise cluster if present
        continue
    cluster_searches = kmeans_counts[kmeans_counts['kmeans_label'] == label]['search_text'].tolist()
    kmeans_ai_names[label] = get_cluster_name(cluster_searches)


# Map AI names back to main dataframe
search_counts['kmeans_name'] = search_counts['kmeans_label'].map(kmeans_ai_names).fillna('Noise')

# reduce dimensionality
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
proj = reducer.fit_transform(embeddings)
search_counts['x'], search_counts['y'] = proj[:,0], proj[:,1]



# display data
st.title('Colorado Catholic Business Directory')
st.write('Data from {} to {}'.format(slider_start, slider_end))


st.divider()
st.header('High Level Metrics')
with st.container():
    st.subheader('Daily Active Users')
    st.line_chart(dau, x='the_date', y='dau', x_label='Date', y_label='DAU')

with st.container():
    st.subheader('Monthly Active Users')
    st.line_chart(mau, x='month_year', y='monthly_active_users', x_label='Month', y_label='MAU')

with st.container():
    st.subheader('Page Views')
    st.line_chart(pvs, x='the_date', y='page_views', x_label='Date', y_label='Page Views')

with st.container():
    st.subheader('Average Page Views per User')
    st.line_chart(page_views_per_user, x='the_date', y='avg_page_views_per_user', x_label='Date', y_label='Page Views per User')



st.divider()
st.header('User Referrers and Platforms')
col1, col2 = st.columns(2)
with st.container():
    col1.subheader('Referrers')
    col1.bar_chart(referrers, x='referrer', y='count', x_label='Referrer', y_label='Count', horizontal=True, use_container_width=True)

with st.container():
    col2.subheader('Platforms')
    col2.bar_chart(platform_mobile, x='platform', y='count', x_label='Platform', y_label='Count', horizontal=True, use_container_width=True)


st.divider()
st.header('User Behavior')
with st.container():
    st.subheader('Average Number of Searches per User')
    st.line_chart(search_results, x='the_date', y='avg_business_count', x_label='Date', y_label='Average Number of Searches per User')

with st.container():
    st.subheader('Average Number of Impressions per Business')
    st.line_chart(impressions, x='the_date', y='avg_impressions', color='sponsored_listing', x_label='Date', y_label='Average Number of Impressions per Business')

with st.container():
    st.subheader('Average Number of Page Views per Business')
    st.line_chart(page_views_per_business, x='the_date', y='avg_page_views', x_label='Date', y_label='Average Number of Page Views per Business')

with st.container():
    st.subheader('Average Number of Visit Website Button Clicks per Business')
    st.line_chart(button_clicks, x='the_date', y='avg_button_clicks', x_label='Date', y_label='Average Number of Visit Website Button Clicks per Business')

with st.container():
    st.subheader('Daily Signups')
    st.line_chart(signups, x='month_year', y='signups', x_label='Month', y_label='Signups')




st.divider()
st.header('Search Clustering')
st.write('This is a scatter plot of the search terms and their clusters. The clusters are generated by a KMeans clustering algorithm.')
st.write('Use the Number of Clusters slider to change the number of clusters.')
st.scatter_chart(search_counts, x='x', y='y', color='kmeans_name', x_label='UMAP X', y_label='UMAP Y')