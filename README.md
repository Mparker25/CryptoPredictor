# Crypto Predictor

Following this [Tutorial](https://towardsdatascience.com/predict-any-cryptocurrency-applying-nlp-with-global-news-e938af6f7922)

# Difference of Logs

Using Difference of Logs can tell you percentage changes
```python
# percent_diff = log('price') - log('open')
df['log_diff'] = np.log(df['price']) - np.log(df['open'])
```

Track percentage changes over each day and tell how well your crypto is doing.

This alone doesn't help you predict but maybe you can make a model that can precict based on twitter news data


## Side-Note

Mid-way through this project, I found an interesting story on intel as I see on the TV that Intel dropped 20% on it's Data Center Sales.
[Why You Should Short Intel](https://seekingalpha.com/article/4418015-intel-short-this-fab-play)

TLDR: Intel is building 2 new fabs in Arizona for $20 Billion. They are creating 10nm chips while everyone is moving onto 3nm chips. Intel is roughly a decade behind and plans on speninding large swathes of cash to play catchup, whilst also losing revenue. They are gambling against $20 Billion Upkeep and diminishing profits vs regaining traction and profit. I'm interested in learning if shorting Intel is a smart move and how to do it!
