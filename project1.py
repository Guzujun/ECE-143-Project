# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# %%
# load the dataset
import zipfile
import kagglehub
import os

path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
print("Path to dataset files:", path)

recipes_path = os.path.join(path, 'recipes.csv')
reviews_path = os.path.join(path, 'reviews.csv')
recipes = pd.read_csv(recipes_path)
reviews_df = pd.read_csv(reviews_path)
recipes.head()


# %%
# load the dataset
import zipfile

path = "data.zip"
with zipfile.ZipFile(path, "r") as z:
    file_list = z.namelist()
    print("ZIP:", file_list)
    # load the recipes data
    with z.open("recipes.csv") as f:
        recipes = pd.read_csv(f)
recipes.head()


# %% [markdown]
# ## **1. Data Exploration**

# %%
print(recipes.info())

# %%
print(recipes.isnull().sum())

# %%
## select those columns we care about
selected_columns = ['RecipeId','Name','Calories', 'FatContent', 'SaturatedFatContent','CholesterolContent', 'SodiumContent', 'CarbohydrateContent','FiberContent', 'SugarContent', 'ProteinContent','RecipeIngredientParts','RecipeInstructions']

data = recipes[selected_columns]

# %%
data.head()

# %%
print(data.isnull().sum())


# %%
fields_of_interest = [
    "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
    "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent",
    "ProteinContent"
]

plt.figure(figsize=(15, 10))
for i, field in enumerate(fields_of_interest, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=data[field])
    plt.title(f"Boxplot of {field}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### We can see from BoxPlot that the data is **serverely affected by the outlier**

# %%
# remove the outlier by IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)][column]

## plot again
plt.figure(figsize=(15, 10))
for i, field in enumerate(fields_of_interest, 1):
    filtered_data = remove_outliers(data, field)
    plt.subplot(3, 3, i)
    sns.boxplot(filtered_data)
    plt.title(f"Boxplot of {field}")
plt.tight_layout()
plt.show()


# %%
# plot distribution

plt.figure(figsize=(15, 10))
for i, field in enumerate(fields_of_interest, 1):
    plt.subplot(3, 3, i)
    filtered_data = remove_outliers(data, field)
    plt.hist(filtered_data, bins=50, edgecolor="black", alpha=0.7)
    plt.title(f"Distribution of {field}")
    plt.xlabel(field)
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# %%
recipes['DatePublished'] = pd.to_datetime(recipes['DatePublished'], errors='coerce')
recipes["YearPublished"] = recipes["DatePublished"].dt.year.astype("Int64")
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)][["YearPublished", column]]

plt.figure(figsize=(15, 10))
for i, field in enumerate(fields_of_interest, 1):
    plt.subplot(3, 3, i)
    field_data = remove_outliers(recipes, field)
    yearly_average = field_data.groupby('YearPublished')[field].mean()
    sns.lineplot(x=yearly_average.index, y=yearly_average, marker='o', label='Average')
    plt.title(f"Change in {field} Over Years")
    plt.xlabel('Year Published')
    plt.ylabel('Average ' + field)
plt.tight_layout()
plt.show()

# %%

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, 9))
nutrients = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
recipes["RoundedRating"] = recipes["AggregatedRating"].round()
average_nutrition = recipes.groupby("RoundedRating")[nutrients].mean()
for nutrient, color in zip(nutrients, colors):
    ax.plot(average_nutrition.index, average_nutrition[nutrient], marker='o', label=nutrient, color=color, linestyle='-', linewidth=2)

ax.set_xlabel('Average Rating (Rounded)')
ax.set_ylabel('Amount')
ax.set_title('Nutritional Content by Average Rating')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
plt.subplots_adjust(bottom=0.25)

plt.show()


# %%
from wordcloud import WordCloud

five_star_recipes = reviews_df[reviews_df["Rating"] == 5]
reviews = " ".join(five_star_recipes["Review"].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of 5 Star Recipe Reviews")
plt.show()

# %%
ingredients = " ".join(recipes["RecipeIngredientParts"].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(ingredients)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Common Recipe Ingredients")
plt.show()

# %%
plt.figure(figsize=(12, 6))
sns.heatmap(recipes[nutrients].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Nutritional Features")
plt.show()

# %% [markdown]
# ### **Data Distribution Analysis**
# 
# #### **Skewed Distribution (Right-Skewed / Long-Tailed Distribution)**  
# - **Fat Content (`FatContent`), Carbohydrates (`CarbohydrateContent`), and Protein (`ProteinContent`)** exhibit a **right-skewed distribution**, where most values are concentrated in a lower range, but some extend significantly higher.  
# - This indicates that **most recipes have relatively low nutritional values**, but there are still some **high-calorie, high-fat, or high-protein** recipes.
# 
# #### **Cholesterol (`CholesterolContent`) and Sodium (`SodiumContent`)**  
# - **Cholesterol (`CholesterolContent`)**: The majority of recipes contain little to no cholesterol, but some have significantly higher values, likely from **high-cholesterol foods such as meat, egg yolks, etc.**  
# - **Sodium (`SodiumContent`)**: This also follows a long-tail distribution, meaning **most recipes have low sodium content**, but a few contain **very high sodium levels**, typically **processed or preserved foods**.
# 
# #### **Sugar (`SugarContent`) and Fiber (`FiberContent`)**  
# - **Sugar (`SugarContent`) and Fiber (`FiberContent`)** both show a **right-skewed distribution**, which aligns with expectations.  
# - **Most recipes contain low sugar**, but some, such as **desserts and beverages,** have high sugar content.  
# - **Fiber content is relatively low**, suggesting that many recipes might be made from **refined foods or meats**.
# 
# #### **Calories (`Calories`)**  
# - The overall distribution appears **reasonable**, with most recipes containing **0-600 calories**, which is typical for nutritional distribution.  
# - However, some **high-calorie recipes** still exist, likely corresponding to **cakes, fast food, or other energy-dense meals**.

# %% [markdown]
# ## **2. Build the Recommendation System**

# %%
## import libaries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# %%
fields_of_interest = [
    "Calories","FatContent", "SaturatedFatContent", "CholesterolContent",
    "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent",
    "ProteinContent"
]


def preprocess_data(df):
    df = df.copy()
    # remove the outliers
    for col in fields_of_interest:
        df[col] = remove_outliers(df, col)
    # fill in the NA value
    df[fields_of_interest] = df[fields_of_interest].fillna(df[fields_of_interest].median())

    return df


# %%
# recommendation function
# consider there are large amount of data, directly compute the whole similarity matrix could be time-consuming
# use KNN to facilitate the search
def recommend_recipes_knn(df, user_preferences, top_n=5):
    """
    Based on user's nutrients need, recommend the most similar recipe
    user_preferences: dict with user's targeted nutrient, e.g:
    {
        "Calories": 0.3,
        "FatContent": 0.2,
        "ProteinContent": 0.8,
        "SugarContent": 0.1
    }
    """
    # convert into DataFrame
    user_df = pd.DataFrame([user_preferences])

    # scaling
    scaler = StandardScaler()
    user_df[fields_of_interest] = scaler.fit_transform(user_df[fields_of_interest])


    # train the KNN model(with cosine similarity)
    knn = NearestNeighbors(n_neighbors=top_n, metric="cosine")
    knn.fit(df[fields_of_interest])

    # find the most common recipes
    distances, indices = knn.kneighbors(user_df)

    # return recommended recipes
    return df.iloc[indices[0]][["Name"] + fields_of_interest]

# %%
data = preprocess_data(data)

# %%
## test case
user_preferences = {
    "Calories": 500,
    "FatContent": 0.2,
    "SaturatedFatContent": 0.1,
    "CholesterolContent": 0.1,
    "SodiumContent": 0.2,
    "CarbohydrateContent": 0.3,
    "FiberContent": 0.5,
    "SugarContent": 0.1,
    "ProteinContent": 0.8
}
recommended_recipes = recommend_recipes_knn(data, user_preferences, top_n=5)
## recommend based on KNN with cosine similarity
recommended_recipes

# %% [markdown]
# ### Consider the dataset is large(over 500K entries), we can also use **Faiss** to expedite the searching process

# %%
import faiss

# construct faiss index
d = len(fields_of_interest)  # dimension
index = faiss.IndexFlatL2(d)  # recommned based on L2 distance
index.add(np.array(recipes[fields_of_interest], dtype=np.float32))

# find the neareast top 5 recipe
D, I = index.search(np.array([list(user_preferences.values())], dtype=np.float32), 5)

# result
recommended_recipes = data.iloc[I[0]]
# recommend based on faiss with L2 distance
recommended_recipes


# %% [markdown]
# #### The above recommendation systems are not that straightforward:
# #### Users usually don't know what's the nutrient in their food
# ## **--- A MORE DIRECT RECOMMENDATION SYS ---**
# ### Recommend based on the the Calories need and the ingredients
# ### **Design on Recommend System**
# ### Users input their height, weight, health target and the ingredients -- the system will return the TOP K recipes with least Calories based on the gradients

# %%
# calculate the calories user body information and the health goal they hold
def calculate_calories(height, weight, goal):
    bmi = weight / (height ** 2)  # BMI
    tdee = weight * 22 * 1.2  # compute BMR

    # adjust the calories based on the goal
    if goal == "fat_loss":
        tdee -= 500  # loss weight: lessen 500 kcal
    elif goal == "muscle_gain":
        tdee += 300  # gain weight: add 300 kcal
    return max(tdee, 1200)  # maintain the calories above the threshold

# %%
# import re
# from collections import Counter

# def count_word_occurrences(instructions_list):
#     word_counter = Counter()

#     for instructions in instructions_list:
#         words = re.findall(r'\b\w+\b', str(instructions).lower())  # 提取单词
#         word_counter.update(words)

#     return word_counter

# instructions_list = recipes["RecipeInstructions"].to_numpy()
# word_counter = count_word_occurrences(instructions_list)

# %%
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## re preprocess the recipes["RecipeInstructions"]
instructions_list = recipes["RecipeIngredientParts"].to_numpy()

def preprocess_instructions(instructions_list):
    return [set(re.findall(r'\b\w+\b', str(instructions).lower())) for instructions in instructions_list]

preprocessed_instructions = preprocess_instructions(instructions_list)

# %%
# compute the similarity between user's ingredients and recipes' ingredients
def compute_ingredient_similarity(preprocessed_instructions, user_ingredients):
    user_ingredients_str = " ".join(user_ingredients)

    instructions_list = [" ".join(ingredients) for ingredients in preprocessed_instructions]

    #  CountVectorizer + cosine similarity derive the result
    vectorizer = CountVectorizer()
    text_matrix = vectorizer.fit_transform([user_ingredients_str] + instructions_list)
    similarity_scores = cosine_similarity(text_matrix[0], text_matrix[1:]).flatten()

    return similarity_scores

# %%

def recommend_recipes(height, weight, goal, ingredients, recipes, top_n=5):
    # caculate the calories
    target_calories = calculate_calories(height, weight, goal)

    # compute similarity
    similarity_scores = compute_ingredient_similarity(preprocessed_instructions, ingredients)
    data["IngredientSimilarity"] = similarity_scores

    data["CalorieDiff"] = np.abs(data["Calories"] - target_calories)

    # filter the data
    filtered_df = recipes.sort_values(by=["IngredientSimilarity", "CalorieDiff"], ascending=[False, True]).head(top_n)


    if filtered_df.empty:
        return "No matching recipes found. Try different ingredients."

    # return TOP K recipes with least calories based on specific ingredients
    return filtered_df


### Test case
height = 1.75  # height(m)
weight = 65    # weight(kg)
goal = "muscle_gain"  # health target："fat_loss", "muscle_gain", "maintain"
ingredients = {"chicken","carrot","potato"}  # ingredients that user has

# recommend recipes
recommended_recipes = recommend_recipes(height, weight, goal, ingredients, data, top_n=10)


# %% [markdown]
# ## **2. Check the reviews of recipes**

# %%
path = "data.zip"
with zipfile.ZipFile(path, "r") as z:
    file_list = z.namelist()
    print("ZIP:", file_list)
    # load the reviews data
    with z.open("reviews.csv") as f:
        review = pd.read_csv(f)

# %%
selected = review[review["RecipeId"].isin(recommended_recipes["RecipeId"])]
ratings_per_recipe = selected.groupby("RecipeId")["Rating"].apply(list).head(5)
ratings_per_recipe = selected.groupby("RecipeId")["Rating"].agg(['mean', 'count'])

# Calculate a weighted score (you can adjust the factor to balance between the two metrics)
ratings_per_recipe['weighted_score'] = ratings_per_recipe['mean'] * np.log(ratings_per_recipe['count'])
top_3_recipes = ratings_per_recipe.sort_values(by='weighted_score', ascending=False).head(3)
top_3_recipes


# %%
# Trace back the top recipe
top_recipes = ratings_per_recipe.sort_values(by='weighted_score', ascending=False).head(1)
top_recipe_ids = top_3_recipes.index[0]
top_recipes = recommended_recipes[recommended_recipes['RecipeId']==top_recipe_ids]
top_recipes_id = top_recipes["RecipeId"]
top_recipes


# %%
top_reviews = review[review['RecipeId'].isin(top_recipes_id)]
top_reviews

# %%


# %%
top_reviews = review[review['RecipeId'].isin(top_recipes_id)]
from sklearn.feature_extraction.text import CountVectorizer

# Vectorize the comments and get word frequencies
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(top_reviews['Review'])

# Get the word frequency counts
word_frequencies = X.toarray().sum(axis=0)
words = vectorizer.get_feature_names_out()

# Create a DataFrame to display words and their frequencies
word_freq_df = pd.DataFrame(zip(words, word_frequencies), columns=['Word', 'Frequency'])
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

# Display the top keywords
word_freq_df.head(10)

# %%
import pandas as pd
import numpy as np
from textblob import TextBlob
import re

def analyze_review_characteristics(reviews_df):
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return text
    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity

    def detect_suggestion_score(text):
        suggestion_keywords = [
            'would', 'could', 'might', 'suggest', 'recommend',
            'try', 'maybe', 'perhaps', 'alternative', 'modify',
            'change', 'adjust', 'swap', 'replace', 'improve'
        ]

        text_lower = str(text).lower()

        suggestion_count = sum(1 for keyword in suggestion_keywords if keyword in text_lower)

        has_potential_suggestion = any([
            'next time' in text_lower,
            'i would' in text_lower,
            'i think' in text_lower
        ])

        suggestion_score = suggestion_count + (2 if has_potential_suggestion else 0)

        return suggestion_score

    reviews_df['Cleaned_Review'] = reviews_df['Review'].apply(clean_text)
    reviews_df['Sentiment_Score'] = reviews_df['Review'].apply(get_sentiment)
    reviews_df['Suggestion_Score'] = reviews_df['Review'].apply(detect_suggestion_score)

    most_positive = reviews_df.loc[reviews_df['Sentiment_Score'].idxmax()]
    most_suggestive = reviews_df.loc[reviews_df['Suggestion_Score'].idxmax()]

    # Prepare results
    results = {
        'Most Positive Review': {
            'Text': most_positive['Review'],
            'Sentiment Score': most_positive['Sentiment_Score']
        },
        'Most Suggestive Review': {
            'Text': most_suggestive['Review'],
            'Suggestion Score': most_suggestive['Suggestion_Score']
        }
    }

    # Detailed Printout
    print("Review Analysis Results:")
    print("\n1. Most Positive Review:")
    print(f"   Sentiment Score: {results['Most Positive Review']['Sentiment Score']:.2f}")
    print(f"   Review: {results['Most Positive Review']['Text']}")

    print("\n2. Most Suggestive Review:")
    print(f"   Suggestion Score: {results['Most Suggestive Review']['Suggestion Score']}")
    print(f"   Review: {results['Most Suggestive Review']['Text']}")

    return results, reviews_df

# %%
review_insights, annotated_reviews = analyze_review_characteristics(top_reviews)

# You can then access specific insights
most_positive_review = review_insights['Most Positive Review']['Text']
most_suggestive_review = review_insights['Most Suggestive Review']['Text']

# %%


# %%



