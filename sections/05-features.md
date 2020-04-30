[<<< Previous](04-data.md) | [Next >>>](06-supervised.md)

# Extracting Features

## Defining Features

What should we use as features for our data set?  What did we use as features for our fruit example [before](03-classification.md)?

| Object | Height | Width | Color  | Mass | Round?
| :--:   | :--:   | :--:  | :--:   | :--: | :--:
| Apple  | 6cm    | 7cm   | Red    | 330g | `True`
| Orange | 6cm    | 7cm   | Orange | 330g | `True`
| Lemon  | 5cm    | 4cm   | Yellow | 150g | `False`

Now that we are using sentences, how can we best represent each sentence as a series of values?

One idea is to count how many particular *parts of speech* the sentence contains. In particular, let's see if we can find out how many nouns and adjectives are used in each sentence across our dataset:

- **Nouns**: Most basically described as a person, place, or thing.  Counting nouns can help determine how many topics are being discussed in a sentence.
- **Adjectives**: Descriptors of nouns (e.g. "yellow", "angry", "charming").  Counting adjectives can help determine how often descriptive words are being added to nouns, which can demonstrate writing style.

## Parts of Speech (POS)

Let us first take a look at all of the parts of speech (POS) on each sentence in our dataframe. The sentences are located in the column `sentence`, and to get the parts of speech, we can use the function `pos_tag_sents` from the NLTK package:

```python
pos_all = pos_tag_sents(df['sentence'])
```

Let's look at the first five results:

```python
print (pos_all[:5])
```

```
[[('The', 'DT'), ('Fulton', 'NNP'), ('County', 'NNP'), ('Grand', 'NNP'), ('Jury', 'NNP'), ('said', 'VBD'), ('Friday', 'NNP'), ('an', 'DT'), ('investigation', 'NN'), ('of', 'IN'), ("Atlanta's", 'NNP'), ('recent', 'JJ'), ('primary', 'JJ'), ('election', 'NN'), ('produced', 'VBD'), ('``', '``'), ('no', 'DT'), ('evidence', 'NN'), ("''", "''"), ('that', 'IN'), ('any', 'DT'), ('irregularities', 'NNS'), ('took', 'VBD'), ('place', 'NN'), ('.', '.')], [('The', 'DT'), ('jury', 'NN'), ('further', 'RB'), ('said', 'VBD'), ('in', 'IN'), ('term-end', 'JJ'), ('presentments', 'NNS'), ('that', 'IN'), ('the', 'DT'), ('City', 'NNP'), ('Executive', 'NNP'), ('Committee', 'NNP'), (',', ','), ('which', 'WDT'), ('had', 'VBD'), ('over-all', 'JJ'), ('charge', 'NN'), ('of', 'IN'), ('the', 'DT'), ('election', 'NN'), (',', ','), ('``', '``'), ('deserves', 'VBZ'), ('the', 'DT'), ('praise', 'NN'), ('and', 'CC'), ('thanks', 'NNS'), ('of', 'IN'), ('the', 'DT'), ('City', 'NNP'), ('of', 'IN'), ('Atlanta', 'NNP'), ("''", "''"), ('for', 'IN'), ('the', 'DT'), ('manner', 'NN'), ('in', 'IN'), ('which', 'WDT'), ('the', 'DT'), ('election', 'NN'), ('was', 'VBD'), ('conducted', 'VBN'), ('.', '.')], [('The', 'DT'), ('September-October', 'NNP'), ('term', 'NN'), ('jury', 'NN'), ('had', 'VBD'), ('been', 'VBN'), ('charged', 'VBN'), ('by', 'IN'), ('Fulton', 'NNP'), ('Superior', 'NNP'), ('Court', 'NNP'), ('Judge', 'NNP'), ('Durwood', 'NNP'), ('Pye', 'NNP'), ('to', 'TO'), ('investigate', 'VB'), ('reports', 'NNS'), ('of', 'IN'), ('possible', 'JJ'), ('``', '``'), ('irregularities', 'NNS'), ("''", "''"), ('in', 'IN'), ('the', 'DT'), ('hard-fought', 'JJ'), ('primary', 'NN'), ('which', 'WDT'), ('was', 'VBD'), ('won', 'VBN'), ('by', 'IN'), ('Mayor-nominate', 'NNP'), ('Ivan', 'NNP'), ('Allen', 'NNP'), ('Jr.', 'NNP'), ('.', '.')], [('``', '``'), ('Only', 'RB'), ('a', 'DT'), ('relative', 'JJ'), ('handful', 'NN'), ('of', 'IN'), ('such', 'JJ'), ('reports', 'NNS'), ('was', 'VBD'), ('received', 'VBN'), ("''", "''"), (',', ','), ('the', 'DT'), ('jury', 'NN'), ('said', 'VBD'), (',', ','), ('``', '``'), ('considering', 'VBG'), ('the', 'DT'), ('widespread', 'JJ'), ('interest', 'NN'), ('in', 'IN'), ('the', 'DT'), ('election', 'NN'), (',', ','), ('the', 'DT'), ('number', 'NN'), ('of', 'IN'), ('voters', 'NNS'), ('and', 'CC'), ('the', 'DT'), ('size', 'NN'), ('of', 'IN'), ('this', 'DT'), ('city', 'NN'), ("''", "''"), ('.', '.')], [('The', 'DT'), ('jury', 'NN'), ('said', 'VBD'), ('it', 'PRP'), ('did', 'VBD'), ('find', 'VB'), ('that', 'IN'), ('many', 'JJ'), ('of', 'IN'), ("Georgia's", 'NNP'), ('registration', 'NN'), ('and', 'CC'), ('election', 'NN'), ('laws', 'NNS'), ('``', '``'), ('are', 'VBP'), ('outmoded', 'VBN'), ('or', 'CC'), ('inadequate', 'JJ'), ('and', 'CC'), ('often', 'RB'), ('ambiguous', 'JJ'), ("''", "''"), ('.', '.')]]
```

What's with those part of speech labels? They are not very self-explanatory...!

The Penn Tagset, which NLTK uses for its part-of-speech tagger, is not particularly intuitive.  Fortunately, they provide an easy function that allows you to see what the different tags stand for.

```python
nltk.help.upenn_tagset("NN")
nltk.help.upenn_tagset("JJ")
```

```
NN: noun, common, singular or mass
    common-carrier cabbage knuckle-duster Casino afghan shed thermostat
    investment slide humour falloff slick wind hyena override subhumanity
    machinist ...
JJ: adjective or numeral, ordinal
    third ill-mannered pre-war regrettable oiled calamitous first separable
    ectoplasmic battery-powered participatory fourth still-to-be-named
    multilingual multi-disciplinary ...
```

## Calculating our Features

Let's create a function that calculates our features across the dataset for us. In this case, numbers of nouns and adjectives that appear in the sentence)

Now we know the tags for the different parts of speech we want to count in each sentence.  Let's write a function that will count the parts of speech to us, when given a part of speech tagged sentence (such as what we have already in our DataFrame) and the part of speech we want to count (for example, "NN" to count the number of nouns in the sentence).

```python
def countPOS(pos_tag_sent, POS):
    pos_count = 0
    all_pos_counts = []
    for sentence in pos_tag_sent:
        for word in sentence:
            tag = word[1]
            if tag [:2] == POS:
                pos_count = pos_count+1
        all_pos_counts.append(pos_count)
        pos_count = 0
    return all_pos_counts
```

We will now call this function twice, one for each of the parts of speech we are counting.  As we finish counting them, we put the results into the DataFrame, saving us the trouble of having to do so later.

```python
df['NN'] = countPOS(pos_all, 'NN')
df['JJ'] = countPOS(pos_all, "JJ")
```

Let's make sure it all looks OK by looking at the leading five rows by running:

```python
df.head()
```

This should present us with the following table:

|   | label  | sentence | NN | JJ
|---|---|---|---|---|---|---|
| **0** | news  |  [The, Fulton, County, Grand, Jury, said, Frida... | 11 | 2
| **1** | news  |  [The, jury, further, said, in, term-end, prese... | 13 | 2
| **2** | news  |  [The, September-October, term, jury, had, been... | 16 | 2
| **3** | news  |  [``, Only, a, relative, handful, of, such, rep... | 9  | 3
| **4** | news  |  [The, jury, said, it, did, find, that, many, o... | 5  | 3

We can also look at the trailing five rows by running:

```python
df.tail()
```

This should yield a result that looks like this:

|   | label  | sentence | NN | JJ
|---|---|---|---|---|---|---|
| **4426** | romance  |  [Nobody, else, showed, pleasure, ... | 2 | 0
| **4427** | romance  |  [Spike-haired, ,, burly, ,, red-faced, ,, deck... | 9 | 3
| **4428** | romance  |  [``, Hello, ,, boss, '', ,, he, said, ,, and, ... | 2 | 0
| **4429** | romance  |  [``, I, suppose, I, can, never, expect, to, ca... | 3  | 0
| **4430** | romance  |  [``, I'm, afraid, not, '', ... | 1 | 0

It all looks good!

Next, let's take a look at how many features we have in the dataset:

```python
df.groupby('label').sum()
```

Running this should provide us with this table:

|             | NN    | JJ
| :---:       | :---: | :---:
| **label**   |       |
| **news**    | 31593 | 6678
| **romance** | 13821 | 4022

## Saving the Dataframe

Pandas provides an easy function to save your DataFrames to your computer as a `.csv` file, a text file containing all the information separated by commas. The function is called `to_csv`.

```python
df.to_csv("df_news_romance.csv", index=False)
```

Here we export to a file named `df_news_romance.csv` and setting `index` to `False` in order to not export the row names. The result of running this function should be a file in the same directory as your Python script.

[<<< Previous](04-data.md) | [Next >>>](06-supervised.md)
