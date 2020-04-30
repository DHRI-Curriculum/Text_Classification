[<<< Previous](01-introduction.md) | [Next >>>](03-classification.md)

# Installation and Setup

## Importing Packages to Python 3

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

- `nltk`, the Natural Language ToolKit, which will be used for corpora and tools:
  - _`brown`_: The Brown Corpus, a text corpus of American English, split into fifteen different categories.
  - _`pos_tag_sents`_ Part of speech taggers (POS): prebuilt functions that are designed to determine the part of speech of every word in the sentence you give them.
- `pandas as pd`: importing the Pandas toolkit, which we will be using for data processing. We are renaming it `pd` to make the command briefer for us to type each time we use it.
- `matplotlib.pyplot as plt`: We will use MatPlotLib for visualizing our data. We are importing the plotting tools here, and renaming them `plt`.
- `sklearn`: This is the "motor" of the machine learning toolkit that we will be using—the scikit-learn machine learning toolkit.
- Finally, we use the code `%matplotlib inline` to ensure our images display clearly in the Jupyter notebook.

---

Note that you can also download the [Jupyter Notebook](../notebooks/intro_to_ml_with_python.ipynb) for this lesson to follow along.

---

[<<< Previous](01-introduction.md) | [Next >>>](03-classification.md)
