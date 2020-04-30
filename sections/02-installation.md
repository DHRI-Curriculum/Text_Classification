[<<< Previous](01-introduction.md) | [Next >>>](03-classification.md)

# Installation and Setup

## Python 3

- the Natural Language Toolkit
  - We will be using both corpora and tools from this package
- pandas
  - We will use this for data processing
  - Comes with Anaconda
- matplotlib
  - We will use this for visualizing our data
  - Comes with Anaconda
- scikit-learn
  - We will use this for machine learning
  - Comes with Anaconda

## Importing Packages

Let's get started by importing some packages we will need for this workshop.

```python
import nltk
from nltk.corpus import brown
from nltk import pos_tag_sents
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
```

- The Brown Corpus: A text corpus of American English, split into fifteen different categories
- Part of speech taggers (POS): prebuilt functions that are designed to determine the part of speech of every word in the sentence you give them
- `pandas as pd`: importing the Pandas toolkit and renaming it `pd`, to make the command briefer for us to type each time we use it
- `matplotlib.pyplot as plt`: importing plotting tools from matplotlib and renaming them plt
- We use the code `%matplotlib inline` to ensure our images display clearly in the Jupyter notebook.
- `sklearn`: the scikit-learn machine learning toolkit

---

Note that you can also download the [Jupyter Notebook](../notebooks/intro_to_ml_with_python.ipynb) for this lesson to follow along.

---

[<<< Previous](01-introduction.md) | [Next >>>](03-classification.md)
