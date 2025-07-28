import streamlit as st
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime

# import data
df = pd.read_csv('ccbd-data.csv')
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
SELECT DATE(originalTimestamp) as the_date, count(distinct messageId) as page_views
FROM df 
WHERE event_type = 'page'
GROUP BY DATE(originalTimestamp)
"""
pvs = duckdb.sql(query).df()





# display data
st.title('Colorado Catholic Business Directory')
st.write('Data from {} to {}'.format(slider_start, slider_end))


# display data
with st.container():
    st.subheader('Daily Active Users')
    st.line_chart(dau, x='the_date', y='dau', x_label='Date', y_label='DAU')

with st.container():
    st.subheader('Page Views')
    st.line_chart(pvs, x='the_date', y='page_views', x_label='Date', y_label='Page Views')