import streamlit as st
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from streamlit.proto import ButtonGroup_pb2
import umap
from openai import OpenAI
from collections import defaultdict
import tiktoken
import random
import textwrap
from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(layout="wide")

try:
    bill_df = pd.read_csv('user-data.csv')
except:
    bill_df = pd.read_csv('../user-data.csv')

bill_df.rename(columns={'unique id': 'userId', 'Subscription Plan': 'subscription_plan', 'Subscription Renew Date': 'renew_date'}, inplace=True)

event_df = pd.DataFrame()

try:
    data_folder = f'event-data'
    data_folder_list = os.listdir(data_folder)
except:
    data_folder = '../event-data'
    data_folder_list = os.listdir('../event-data')

# import data
for file in data_folder_list:
    if file.endswith('.csv'):
        loop_df = pd.read_csv(f'{data_folder}/{file}')
        event_df = pd.concat([event_df, loop_df])

event_df = event_df[event_df['originalTimestamp'] >= '2025-01-01']
df = event_df.merge(bill_df, on='userId', how='left')


# sidebar menu
data_min_date = datetime.strptime(df['originalTimestamp'].min().split()[0], '%Y-%m-%d')
data_max_date = datetime.strptime(df['originalTimestamp'].max().split()[0], '%Y-%m-%d')

st.sidebar.button("Reset", type="primary")

st.sidebar.header("Date Selectors")
st.sidebar.write("Quick Selectors")
button_group = False
if st.sidebar.button("2025 Q2", width="stretch"):
    button_group = True
    min_date = '2025-04-01'
    max_date = '2025-06-30'
if st.sidebar.button("2025 Q3", width="stretch"):
    button_group = True
    min_date = '2025-07-01'
    max_date = '2025-09-30'
if st.sidebar.button("2025 Q4", width="stretch"):
    min_date = '2025-10-01'
    max_date = '2025-12-31'
    button_group = True
if st.sidebar.button("YTD", width="stretch"):
    button_group = True
    min_date = '2025-01-01'
    max_date = '2025-12-31'
if st.sidebar.button("Last Week", width="stretch"):
    button_group = True
    min_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    max_date = datetime.now().strftime('%Y-%m-%d')


input_date = st.sidebar.date_input(label='Date Range Selector',
    value=(data_min_date, data_max_date),
    min_value=data_min_date,
    max_value=data_max_date,
    format="MM/DD/YYYY",
)

if button_group:
    df = df[df['originalTimestamp'] >= min_date]
    df = df[df['originalTimestamp'] <= max_date]
else:
    start_date = input_date[0].strftime('%Y-%m-%d')
    end_date = input_date[1].strftime('%Y-%m-%d')

    df = df[df['originalTimestamp'] >= start_date]
    df = df[df['originalTimestamp'] <= end_date]


st.sidebar.header('Data Aggregation Level')
agg_type_options = ['day', 'week', 'month']
agg_type = st.sidebar.segmented_control('Aggregation Options', agg_type_options, 
selection_mode="single", default='day')
if agg_type == 'day':
    chart_title = 'Daily'
if agg_type == 'week':
    chart_title = 'Weekly'
if agg_type == 'month':
    chart_title = 'Monthly'

#num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=7)




# data processing
query = f"""
with base as (
    SELECT 
        TRY_CAST(strptime(renew_date, '%b %d, %Y %I:%M %p') AS DATE) AS renew_date,
        subscription_plan,
        userId
    FROM bill_df
)

, agg as (
    select DATE_TRUNC('{agg_type}', renew_date) as the_date,
    subscription_plan, 
    count(distinct userId) as num_customers
    from base
    group by 1, 2
)

select *,
case when subscription_plan = 'monthly' then num_customers*20 else num_customers*200 end as revenue
from agg
"""
rev_calc = duckdb.sql(query).df()

query = """
select the_date,
sum(revenue) as estimated_revenue
from rev_calc
where the_date is not null
group by 1
order by 1
"""
sub_date = duckdb.sql(query).df()


query = f"""
with base as (
    SELECT DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, COALESCE(userId, anonymousId) as user_id
    FROM df 
)

select the_date, count(distinct user_id) as active_users 
from base
group by the_date
order by the_date
""".format(agg_type=agg_type)
active_users = duckdb.sql(query).df()


query = f"""
SELECT DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, count(distinct messageId) as page_views
FROM df 
WHERE event_type = 'page'
GROUP BY DATE_TRUNC('{agg_type}', DATE(originalTimestamp))
"""
pvs = duckdb.sql(query).df()

query = f"""
with base as (
    SELECT DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, 
           COALESCE(userId, anonymousId) as user_id,
           COUNT(DISTINCT messageId) as user_page_views
    FROM df 
    WHERE event_type = 'page'
    GROUP BY DATE_TRUNC('{agg_type}', DATE(originalTimestamp)), COALESCE(userId, anonymousId)
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

query = f"""
with impressions as (
    SELECT DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, 
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

query = f"""
with impressions as (
    SELECT DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, 
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

query = f"""
with t1 as (
    SELECT DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, business_id, count(distinct messageId) as page_views
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

query = f"""
with t1 as (
    SELECT DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, business_id, count(distinct messageId) as button_clicks
    FROM df 
    WHERE event_type = 'button_click'
    and button_name = 'visit_website'
    group by DATE_TRUNC('{agg_type}', DATE(originalTimestamp)), business_id
)

select the_date, avg(button_clicks) as avg_button_clicks
from t1
group by the_date
order by the_date
"""
button_clicks = duckdb.sql(query).df()



query = f"""
with extra as (
    select null as the_date, null as subscription_plan, 31 as signups
)

, base as (
    select DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date, subscription_plan, count(distinct business_id) as signups
    from df
    where event_type = 'button_click'
    and button_name = 'faith'
    group by DATE_TRUNC('{agg_type}', DATE(originalTimestamp)), subscription_plan
)

select * 
from base
union all
select *
from extra
"""
signups = duckdb.sql(query).df()


query = f"""
WITH page_sequence AS (
    SELECT 
        DATE_TRUNC('{agg_type}', DATE(originalTimestamp)) as the_date,
        COALESCE(userId, anonymousId) as user_id,
        name as current_page,
        LEAD(name) OVER (
            PARTITION BY COALESCE(userId, anonymousId) 
            ORDER BY originalTimestamp
        ) as next_page,
        originalTimestamp
    FROM df
    WHERE event_type = 'page'
    AND name IN ('search', 'business_profile')
),
user_weekly_views AS (
    SELECT 
        the_date,
        user_id,
        COUNT(CASE WHEN current_page = 'search' THEN 1 END) as user_search_views,
        COUNT(CASE WHEN current_page = 'business_profile' THEN 1 END) as user_profile_views,
        COUNT(CASE WHEN current_page = 'search' AND next_page = 'business_profile' THEN 1 END)::FLOAT / 
            NULLIF(COUNT(CASE WHEN current_page = 'search' THEN 1 END), 0) as user_ctr
    FROM page_sequence
    GROUP BY the_date, user_id
)
SELECT 
    the_date,
    AVG(user_search_views) as search_views,
    AVG(user_profile_views) as profile_views,
    AVG(user_ctr) as click_through_rate
FROM user_weekly_views
GROUP BY the_date
ORDER BY the_date
"""
ctr = duckdb.sql(query).df()




# query = """
# SELECT lower(search_text) as search_text
# FROM df
# WHERE event_type = 'search'
# and search_text is not null
# """
# search_counts = duckdb.sql(query).df()

# TEXT_COL = 'search_text'
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(
#     search_counts[TEXT_COL].tolist(),
#     batch_size=64,
#     show_progress_bar=True,
#     convert_to_numpy=True,
#     normalize_embeddings=True
# )

# km = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
# km_labels = km.fit_predict(embeddings)
# search_counts['kmeans_label']   = km_labels

# kmeans_counts = search_counts.groupby(['kmeans_label', 'search_text']).size().reset_index(name='count')
# kmeans_counts = kmeans_counts.sort_values(['kmeans_label', 'count'], ascending=[True, False])

# MODEL = "gpt-5-mini"
# client = OpenAI()

# # Helper: ensure prompt stays in model’s token limit
# enc = tiktoken.encoding_for_model(MODEL)
# def too_long(rows, max_tokens=4000):
#     prompt = ' '.join(rows)
#     return (len(enc.encode(prompt)) + 30) > max_tokens

# def get_cluster_name(rows):
#     if too_long(rows):
#         prompt_rows = rows[:20] + rows[-20:]
#     else:
#         prompt_rows = rows

#     user_prompt = f"""Given this data {str(prompt_rows)},
#     what is a good one or two word category for a search term scatter plot to understand what users on a business directory are searching for.
#     Only return the category name."""
    
#     response = client.responses.create(
#         model=MODEL,
#         instructions="You are a coding assistant that only gives one or two word answers.",
#         input=user_prompt
#     )

#     return response.output_text

# # Get AI-generated names for each cluster
# kmeans_ai_names = {}

# # Get KMeans cluster names
# for label in kmeans_counts['kmeans_label'].unique():
#     if label == -1:  # Skip noise cluster if present
#         continue
#     cluster_searches = kmeans_counts[kmeans_counts['kmeans_label'] == label]['search_text'].tolist()
#     kmeans_ai_names[label] = get_cluster_name(cluster_searches)


# # Map AI names back to main dataframe
# search_counts['kmeans_name'] = search_counts['kmeans_label'].map(kmeans_ai_names).fillna('Noise')

# # reduce dimensionality
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
# proj = reducer.fit_transform(embeddings)
# search_counts['x'], search_counts['y'] = proj[:,0], proj[:,1]



# display data
st.title('Colorado Catholic Business Directory')

st.divider()
st.header('User Behavior') 
with st.container():
    total, chart = st.columns(spec=[.2, .8])
    total.metric(label=f'Total Signups', value=signups['signups'].sum())
    total.metric(label=f'Total Non-Paid Signups', value=signups[signups['subscription_plan'].isna()]['signups'].sum())
    total.metric(label=f'Total Monthly Signups', value=signups[signups['subscription_plan'] == 'monthly']['signups'].sum())
    total.metric(label=f'Total Yearly Signups', value=signups[signups['subscription_plan'] == 'yearly']['signups'].sum())
    chart.subheader(f'{chart_title} Signups')
    chart.bar_chart(signups, x='the_date', y='signups', x_label='Date', y_label='Signups', color='subscription_plan')

with st.container():
    total, chart = st.columns(spec=[.2, .8])
    total.metric(label=f'Total Active Users', value=active_users['active_users'].sum())
    chart.subheader(f'{chart_title} Active Users')
    chart.line_chart(active_users, x='the_date', y='active_users', x_label='Date', y_label='Active Users')

with st.container():
    total, chart = st.columns(spec=[.2, .8])
    total.metric(label=f'Total Page Views', value=pvs['page_views'].sum())
    chart.subheader(f'{chart_title} Page Views')
    chart.line_chart(pvs, x='the_date', y='page_views', x_label='Date', y_label='Page Views')

with st.container():
    total, chart = st.columns(spec=[.2, .8])
    total.metric(label=f'Average Page Views Users', value=round(page_views_per_user['avg_page_views_per_user'].mean(), 2))
    chart.subheader(f'{chart_title} Average Page Views per User')
    chart.line_chart(page_views_per_user, x='the_date', y='avg_page_views_per_user', x_label='Date', y_label='Page Views per User')



# st.divider()
# st.header('User Referrers and Platforms')
# col1, col2 = st.columns(2)
# with st.container():
#     col1.subheader('Referrers')
#     col1.bar_chart(referrers, x='referrer', y='count', x_label='Referrer', y_label='Count', horizontal=True, use_container_width=True)

# with st.container():
#     col2.subheader('Platforms')
#     col2.bar_chart(platform_mobile, x='platform', y='count', x_label='Platform', y_label='Count', horizontal=True, use_container_width=True)


# st.divider()
# st.header('User Behavior')
# with st.container():
#     st.subheader('Average Number of Impressions per Business')
#     st.line_chart(impressions, x='the_date', y='avg_impressions', color='sponsored_listing', x_label='Date', y_label='Average Number of Impressions per Business')

# with st.container():
#     st.subheader('Average Number of Page Views per Business')
#     st.line_chart(page_views_per_business, x='the_date', y='avg_page_views', x_label='Date', y_label='Average Number of Page Views per Business')

# with st.container():
#     st.subheader('Average Number of Visit Website Button Clicks per Business')
#     st.line_chart(button_clicks, x='the_date', y='avg_button_clicks', x_label='Date', y_label='Average Number of Visit Website Button Clicks per Business')

# with st.container():
#     st.subheader('Daily Signups')
#     st.line_chart(signups, x='month_year', y='signups', x_label='Month', y_label='Signups')

# with st.container():
#     st.subheader('Search to Profile Click Through Rate')
#     st.line_chart(ctr, x='the_date', y='click_through_rate', x_label='Date', y_label='Search to Profile Click Through Rate')




# st.divider()
# st.header('Search Clustering')
# st.write('This is a scatter plot of the search terms and their clusters. The clusters are generated by a KMeans clustering algorithm.')
# st.write('Use the Number of Clusters slider to change the number of clusters.')
# st.scatter_chart(search_counts, x='x', y='y', color='kmeans_name', x_label='UMAP X', y_label='UMAP Y')

st.divider()
st.header('Revenue Forecast')
st.bar_chart(rev_calc, x='the_date', y='num_customers', color='subscription_plan', x_label='Renew Date', y_label='Number of Paid Customers')

total, chart = st.columns(spec=[.2, .8])
total.metric(label='Total Forecasted Revenue', value=sub_date['estimated_revenue'].sum())
chart.bar_chart(sub_date, x='the_date', y='estimated_revenue', x_label='Renew Date', y_label='Estimated Revenue')

@st.cache_data
def convert_for_download(download_df):
    return download_df.to_csv().encode("utf-8")

csv = convert_for_download(rev_calc)

st.write('You can download the data for the forecast by clicking the button below.')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv",
    icon=":material/download:",
)