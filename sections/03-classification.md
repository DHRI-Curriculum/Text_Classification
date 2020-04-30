[<<< Previous](02-installation.md) | [Next >>>](04-data.md)

# What Is Classification?

Let's show an example of classification using fruit!

## Example: Fruit

How would you describe apples to a computer? How would they differ from oranges?

Remember, computers can only really understand numbers, boolean values (`True`/`False`), and strings within a predefined set.

| Object | Height | Width | Color  | Mass | Round?
| :--:   | :--:   | :--:  | :--:   | :--: | :--:
| Apple  | 6cm    | 7cm   | Red    | 330g | `True`
| Orange | 6cm    | 7cm   | Orange | 330g | `True`
| Lemon  | 5cm    | 4cm   | Yellow | 150g | `False`

Our fruit test shows us everything we need to do a classification machine learning test. For each item with a _label_ (apple, orange, lemon), we use a series of values to try to capture machine-understandable information about the item. These values are a _feature representation_ of the item in question. The features themselves, as we can see above, can be numeric, boolean values (`True`/`False`), or a string in a set of predefined strings.

## Introducing an Unknown Fruit

What if we had a new, unknown fruit?

| Object | Height | Width | Color  | Mass | Round?
| :--:   | :--:   | :--:  | :--:   | :--: | :--:
| Apple  | 6cm    | 7cm   | Red    | 330g | `True`
| Orange | 6cm    | 7cm   | Orange | 330g | `True`
| Lemon  | 5cm    | 4cm   | Yellow | 150g | `False`
| ?????  | 5cm    | 6cm   | Orange | 300g | `True`

Our fruit test is an example of a _classification_ task. Classification allows you to predict a _categorical_ value. This is a type of **supervised machine learning**, meaning we know the labels ahead of time and can give them to the machine learning algorithm so that it can be trained to knows what the categories of our data are. This way, when it comes time to give the algorithm previously unseen data, it knows which categories it's looking for.

We acknowledge that often times we are not trying to divide apples and oranges, but categories of people or personal attributes. While we are going to focus on the mechanics of machine learning, we strongly recommend that this work be paired with a grounding in ethics, such as the [DHRI-Ethics](https://github.com/DHRI-Curriculum/ethics) workshop.

---

###### The fruit examples on this page come from Andrew Rosenberg 2014 class in Methods in Computational Linguistics.

---

[<<< Previous](02-installation.md) | [Next >>>](04-data.md)
