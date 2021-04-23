import pandas as pd
import numpy as np
import OpenBlender
import json


token = open("token.key").readline()

action = 'API_getObservationsFromDataset'
# ANCHOR: 'Bitcoin vs USD'
  
parameters = { 
    'token' : token,
    'id_dataset' : '5d4c3af79516290b01c83f51',
    'date_filter':{"start_date" : "2020-01-01",
                   "end_date" : "2020-08-29"} 
}
df = pd.read_json(json.dumps(OpenBlender.call(action, parameters)['sample']), convert_dates=False, convert_axes=False).sort_values('timestamp', ascending=False)
df.reset_index(drop=True, inplace=True)
df['date'] = [OpenBlender.unixToDate(ts, timezone = 'GMT') for ts in df.timestamp]
df = df.drop('timestamp', axis = 1)

# Calculate the Logarithmic Difference 
df['log_diff'] = np.log(df['price']) - np.log(df['open'])
df['target'] = [1 if log_diff > 0 else 0 for log_diff in df['log_diff']]

# Create a Unix Timestamp
format = '%d-%m-%Y %H:%M:%S'
timezone = 'GMT'
df['timestamp'] = OpenBlender.dateToUnix(df['date'], date_format = format, timezone = timezone)
df = df[['date', 'timestamp', 'price', 'target']]


# Search for Datasets on OpenBlender
search_keyword = 'bitcoin'
df = df.sort_values('timestamp').reset_index(drop = True)
# print('From : ' + OpenBlender.unixToDate(min(df.timestamp)))
# print('Until: ' + OpenBlender.unixToDate(max(df.timestamp)))
OpenBlender.searchTimeBlends(token, df.timestamp, search_keyword)

# We need to add the 'id_dataset' and the 'feature' name we want.
blend_source = {
                'id_dataset':'5ea2039095162936337156c9',
                'feature' : 'text'
            }

# Now, let's 'timeBlend' it to our dataset
df_blend = OpenBlender.timeBlend( token = token,
                                  anchor_ts = df.timestamp,
                                  blend_source = blend_source,
                                  blend_type = 'agg_in_intervals',
                                  interval_size = 60 * 60 * 24,
                                  direction = 'time_prior',
                                  interval_output = 'list',
                                  missing_values = 'raw')

df = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)

# We add the ngrams to match on a 'positive' feature.
positive_filter = {'name' : 'positive', 
                   'match_ngrams': ['positive', 'buy', 
                                    'bull', 'boost']}
blend_source = {
                'id_dataset':'5ea2039095162936337156c9',
                'feature' : 'text',
                'filter_text' : positive_filter
            }
df_blend = OpenBlender.timeBlend( token = token,
                                  anchor_ts = df.timestamp,
                                  blend_source = blend_source,
                                  blend_type = 'agg_in_intervals',
                                  interval_size = 60 * 60 * 24,
                                  direction = 'time_prior',
                                  interval_output = 'list',
                                  missing_values = 'raw')
df = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)

# And now the negatives
negative_filter = {'name' : 'negative', 
                   'match_ngrams': ['negative', 'loss', 'drop', 'plummet', 'sell', 'fundraising']}
blend_source = {
                'id_dataset':'5ea2039095162936337156c9',
                'feature' : 'text',
                'filter_text' : negative_filter
            }
df_blend = OpenBlender.timeBlend( token = token,
                                  anchor_ts = df.timestamp,
                                  blend_source = blend_source,
                                  blend_type = 'agg_in_intervals', #closest_observation
                                  interval_size = 60 * 60 * 24,
                                  direction = 'time_prior',
                                  interval_output = 'list',
                                  missing_values = 'raw')
df = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)

features = ['target', 'BITCOIN_NE.text_COUNT_last1days:positive', 'BITCOIN_NE.text_COUNT_last1days:negative']

# Correlate the Percentage Target with the pos or neg Bitcoin words
df_anchor = df[features].corr()['target']

# Vectorize the text
blend_source = { 
                'id_textVectorizer':'5f739fe7951629649472e167'
               }
df_blend = OpenBlender.timeBlend( token = token,
                                  anchor_ts = df.timestamp,
                                  blend_source = blend_source,
                                  blend_type = 'agg_in_intervals',
                                  interval_size = 60 * 60 * 24,
                                  direction = 'time_prior',
                                  interval_output = 'list',
                                  missing_values = 'raw') .add_prefix('VEC.')
df_anchor = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)
print(df_anchor.head())

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
# We drop correlated features because with so many binary 
# ngram variables there's a lot of noise
corr_matrix = df_anchor.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
df_anchor.drop([column for column in upper.columns if any(upper[column] > 0.5)], axis=1, inplace=True)

# Now we separate in train/test sets
X = df_anchor.loc[:, df_anchor.columns != 'target'].select_dtypes(include=[np.number]).drop(drop_cols, axis = 1).values
y = df_anchor.loc[:,['target']].values
div = int(round(len(X) * 0.2))
X_train = X[:div]
y_train = y[:div]
X_test = X[div:]
y_test = y[div:]
# Finally, we perform ML and see results
rf = RandomForestRegressor(n_estimators = 1000, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
df_res = pd.DataFrame({'y_test':y_test[:, 0], 'y_pred':y_pred})
threshold = 0.5
preds = [1 if val > threshold else 0 for val in df_res['y_pred']]
print(metrics.confusion_matrix(preds, df_res['y_test']))
print('Accuracy Score:')
print(accuracy_score(preds, df_res['y_test']))
print('Precision Score:')
print(precision_score(preds, df_res['y_test']))