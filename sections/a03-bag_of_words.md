[<<< Previous](10-resources.md) | [Back to beginning >>>](../README.md)

# Appendix: Feature Extraction Using Bag of Words

We're almost ready to do some machine learning! First, we need to turn our sentences into the type of _feature vectors_ the algorithm we plan to work with expects. Jumping ahead a bit, the `sklearn` implementation of the algorithm we will use for unsupervised learning requires that the text be in _bag of words_ form, which is the unique words in the text and the count of occurances of that word.

## Read data in from a spreadsheet

Let's take the data we just saved out and load it back into a DataFrame so that we can do some analysis with it!

```python
import pandas as pd
df = pd.read_csv("df_news_romance.csv")
df.head()
```

Our resulting table should look something like this:

|   | label  | sentence | NN | JJ |
|---|---|---|---|---|
| **0** | news  |  ['The', 'Fulton', 'County', 'Grand', 'Jury'... | 11 | 2
| **1** | news  |  ['The', 'jury', 'further', 'said', 'in', 'term'... | 13 | 2
| **2** | news  |  ['The', 'September-October', 'term', 'jury'... | 16 | 2
| **3** | news  |  ['``', 'Only', 'a', 'relative', 'handful', 'of'... | 9 | 3
| **4** | news  |  ['The', 'jury', 'said', 'it', 'did', 'find'... | 5 | 3

Then we print the first 5 rows of the _sentence_ column in the DataFrame:

```python
df['sentence'].head()
```

We should see this:

```
0    ['The', 'Fulton', 'County', 'Grand', 'Jury', '...
1    ['The', 'jury', 'further', 'said', 'in', 'term...
2    ['The', 'September-October', 'term', 'jury', '...
3    ['``', 'Only', 'a', 'relative', 'handful', 'of...
4    ['The', 'jury', 'said', 'it', 'did', 'find', '...
Name: sentence, dtype: object
```

## Bag of Words

We preprocess our data using `sklearn`'s text feature extraction tools. In particular, we use the `CountVectorizer` which computes the frequency of each token in the document. We can strip out _stop words_ (words that are so common they don't add to the data analysis, such as "the" and "a") using the `stop_words` keyword argument. A keyword argument is an optional function parameter.

```python
from sklearn.feature_extraction.text import CountVectorizer

tf_vectorizer = CountVectorizer(stop_words='english')
tf = tf_vectorizer.fit_transform(df['sentence'])
```

`CountVectorizer` processes the text such that `tf` is a sparse matrix containing the count of words in each document. A matrix is a table of numbers, and a sparse matrix is a table where most of those numbers are 0. `tf` is mostly 0 because many words only appear in a handful of the many documents that make up our sample corpus.

One document in the Brown corpus is the following sentence:

> Mrs. Robert O. Spurdle is chairman of the committee , which includes Mrs. James A. Moody , Mrs. Frank C. Wilkinson , Mrs. Ethel Coles , Mrs. Harold G. Lacy , Mrs. Albert W. Terry , Mrs. Henry M. Chance , 2d , Mrs. Robert O. Spurdle , Jr. , Mrs. Harcourt N. Trimble , Jr. , Mrs. John A. Moller , Mrs. Robert Zeising , Mrs. William G. Kilhour , Mrs. Hughes Cauffman , Mrs. John L. Baringer and Mrs. Clyde Newman .

Through the `CountVectorizer` command, the stop words, punctuation, and very low frequency words have been removed. This yeilds the words and their counts, which are listed and also visualized in a word cloud below. The creation of this visualization is discussed in an [appendix](a05-word_cloud.md).

```json
{
  "2d": 1,
  "albert": 1,
  "baringer": 1,
  "cauffman": 1,
  "chairman": 1,
  "chance": 1,
  "clyde": 1,
  "coles": 1,
  "committee": 1,
  "ethel": 1,
  "frank": 1,
  "harcourt": 1,
  "harold": 1,
  "henry": 1,
  "hughes": 1,
  "includes": 1,
  "james": 1,
  "john": 2,
  "jr": 2,
  "kilhour": 1,
  "lacy": 1,
  "moller": 1,
  "moody": 1,
  "mrs": 15,
  "newman": 1,
  "robert": 3,
  "spurdle": 2,
  "terry": 1,
  "trimble": 1,
  "wilkinson": 1,
  "william": 1,
  "zeising": 1
}
```

![Word cloud visualization, where the size of the word is relative to its frequency in a sentence, of "Mrs. Robert O. Spurdle is chairman of the committee , which includes Mrs. James A. Moody , Mrs. Frank C. Wilkinson , Mrs. Ethel Coles , Mrs. Harold G. Lacy , Mrs. Albert W. Terry , Mrs. Henry M. Chance , 2d , Mrs. Robert O. Spurdle , Jr. , Mrs. Harcourt N. Trimble , Jr. , Mrs. John A. Moller , Mrs. Robert Zeising , Mrs. William G. Kilhour , Mrs. Hughes Cauffman , Mrs. John L. Baringer and Mrs. Clyde Newman ."](images/countvect_wordcloud.png?)

[<<< Previous](10-resources.md) | [Back to beginning >>>](../README.md)
