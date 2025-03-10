## Personalized Diet Recommender System

### Project Overview

The **Personalized Diet Recommender System** is designed to help individuals make healthier dietary choices by providing personalized meal recommendations. Using a content-based filtering approach, the system matches user dietary preferences and nutritional needs with an extensive collection of food recipes. This allows users to receive tailored meal suggestions based on their health goals, such as weight loss, muscle gain, or balanced nutrition.

### Features

- **Personalized Recommendations:** Uses content-based filtering with cosine similarity to suggest meals based on user preferences.

- **Nutritional Analysis:** Extracts key nutritional information from recipes, including calories, fat, protein, carbohydrates, and more.

- **User Ratings and Reviews:** Leverages user feedback to enhance recommendation accuracy.

- **Diverse Recipe Collection:** Provides access to a rich database of over 500,000 recipes across various categories.

### Dataset

This project utilizes the **Food.com - Recipes and Reviews dataset**, which consists of:

#### Recipes Dataset:

- 522,517 unique recipes

- 312 distinct recipe categories

- Detailed attributes, including preparation time, cooking time, ingredients, and step-by-step instructions

- Nutritional data such as calories, fat, sodium, carbohydrates, fiber, sugar, and protein content

#### Reviews Dataset:

- 1,401,982 user reviews

- 271,907 unique users

- Review ratings and text-based feedback on recipes

#### Dataset Source

The dataset is sourced from Kaggle and can be accessed via the following link:
[Kaggle: Food.com - Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

### Real-World Applications

This system can be utilized by:

- **Individuals** seeking customized meal planning for specific dietary goals

- **Healthcare professionals** designing personalized meal plans for patients

- **Restaurants and meal delivery services** to offer tailored menu recommendations

- **Fitness enthusiasts** optimizing their nutrition for performance

## 📂 Project Structure

```
project/
│── project.ipynb       # Jupyter Notebook implementing the recommendation system
│── data/               # Directory containing dataset files (if extracted)
│── README.md           # Documentation file
│── data.zip            # Compressed dataset file (if downloading manually)
```

## 🚀 How to Run the Project

### Prerequisites

Ensure you have Python (>=3.7) installed. You will also need to install Jupyter Notebook to run the `.ipynb` file.

### Steps to Run:

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Set Up the Environment** (Optional but recommended)
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**
   Recommend to download the dependencies mentioned in *Dependencies* section.

4. **Download the Dataset**
   The dataset is automatically downloaded using KaggleHub:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
   ```

   If the dataset is provided as `data.zip`, extract it manually:
   ```bash
   unzip data.zip -d data
   ```

5. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook project.ipynb
   ```
   Open the notebook and execute the cells to process data and generate recommendations.

## 🛠️ Dependencies

This project uses the following third-party Python libraries:

- `numpy` - Numerical computations
- `pandas` - Data handling and manipulation
- `matplotlib` - Visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning utilities
- `kagglehub` - Downloading datasets from Kaggle
- `zipfile` - Handling compressed dataset files

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

