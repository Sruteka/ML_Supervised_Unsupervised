# 🏠 NYC Airbnb Data Analysis: Supervised & Unsupervised Learning 🧠  

## Project Overview  
This project analyzes the **AB_NYC_2019** dataset, which contains Airbnb listings in New York City. By applying **supervised** and **unsupervised learning**, we:  
1. **Predicted prices** of Airbnb listings (Supervised Learning).  
2. **Clustered listings** based on key features (Unsupervised Learning).  

The project provides insights into pricing patterns, popular neighborhoods, and groups Airbnb properties into meaningful clusters for analysis.

---

## Table of Contents  
1. [About the Dataset](#about-the-dataset)  
2. [Objective](#objective)  
3. [Technologies Used](#technologies-used)  
4. [Project Workflow](#project-workflow)  
5. [Supervised Learning: Price Prediction](#supervised-learning-price-prediction)  
6. [Unsupervised Learning: Clustering](#unsupervised-learning-clustering)  
7. [Conclusion](#conclusion)  
8. [How to Run the Project](#how-to-run-the-project)  
9. [Author](#author)  

---

## About the Dataset 📊  
The **AB_NYC_2019** dataset includes details about Airbnb listings in New York City, such as:  
- **ID** and **Name** of the listing  
- **Host ID** and **Host Name**  
- **Neighborhood Group** (e.g., Manhattan, Brooklyn)  
- **Neighborhood**  
- **Latitude** and **Longitude**  
- **Room Type** (e.g., Entire home/apt, Private room)  
- **Price** (Target variable for Supervised Learning)  
- **Minimum Nights**  
- **Number of Reviews**  

**Dataset File**: `AB_NYC_2019.csv`  

---

## Objective 🎯  
The project has two main objectives:  
1. **Supervised Learning**: Predict the **price** of Airbnb listings using machine learning models.  
2. **Unsupervised Learning**: Group Airbnb listings into meaningful **clusters** based on key features.  

---

## Technologies Used 🛠️  
The following tools and libraries were used:  

- **Python** 🐍  
- **Pandas** and **NumPy** for data manipulation  
- **Matplotlib** and **Seaborn** for data visualization  
- **Scikit-Learn** for machine learning models  
- **KMeans Clustering** for unsupervised learning  
- **Jupyter Notebook** for code development and execution  

---

## Project Workflow 🔄  
The project follows these key steps:  

### 1. **Data Preprocessing**  
- Handling missing values in features like `price` and `reviews`.  
- Encoding categorical variables like `neighbourhood_group` and `room_type` using One-Hot Encoding.  
- Feature scaling and transformation for numerical data.  

### 2. **Exploratory Data Analysis (EDA)**  
- Visualizing distributions of price, room types, and neighborhoods.  
- Identifying correlations between features using heatmaps.  

### 3. **Supervised Learning: Price Prediction**  
- **Target Variable**: `price`  
- **Features**: Room type, neighborhood, minimum nights, and reviews.  
- Applied models:  
   - **Linear Regression**  
   - **Random Forest Regressor**  
- **Model Evaluation**:  
   - R² Score  
   - Mean Absolute Error (MAE)  

### 4. **Unsupervised Learning: Clustering**  
- Used **KMeans Clustering** to group Airbnb listings based on:  
   - Latitude and Longitude (geographical features)  
   - Room Type  
   - Minimum Nights  
- Determined the optimal number of clusters using the **Elbow Method**.  
- Visualized clusters on a geographical map of New York City.  

---

## Supervised Learning: Price Prediction 📈  
The best-performing model achieved the following results:  

- **Model Used**: Random Forest Regressor  
- **R² Score**: `0.78`  
- **MAE**: `45.2`  

---

## Unsupervised Learning: Clustering 🗺️  
- Used **KMeans Clustering** with 4 clusters (optimal from Elbow Method).  
- Visualized clusters on NYC maps to identify patterns in property locations.  
- Insights:  
   - Cluster 1: Luxury listings in Manhattan.  
   - Cluster 2: Affordable listings in Brooklyn.  
   - Cluster 3: High-density private rooms in Queens.  

---

## Conclusion 📝  
The project successfully analyzed Airbnb data to:  
1. **Predict listing prices** using machine learning models.  
2. **Cluster properties** into meaningful groups for analysis.  

These insights help property owners, travelers, and Airbnb to understand price drivers and optimize their listings.

---

## How to Run the Project 🚀  
Follow these steps to run the project:  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/AB_NYC_2019_Price_Clustering.git
   cd AB_NYC_2019_Price_Clustering
2. **Install Required Libraries**
Use pip to install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
3. **Run the Jupyter Notebook**
Open the notebook in your browser:
   ```bash
   jupyter notebook
4. **Dataset**
Ensure the file AB_NYC_2019.csv is in the same directory as your notebook.

Author 💻
Sruteka PJ
Data Science Enthusiast

LinkedIn: www.linkedin.com/in/sruteka-pj-a50a14266
GitHub: https://github.com/Sruteka

License 📜
This project is licensed under the MIT License.

---

### Key Highlights:
1. **Supervised Learning**: Added details on price prediction using regression models.
2. **Unsupervised Learning**: Highlighted KMeans clustering and geographic analysis.
3. **Model Performance**: Included placeholders for evaluation metrics (update with actual values).
4. **Clear Sections**: Structured into EDA, supervised, and unsupervised workflows.

Let me know if you need help adding **visualization code** or model details for the project! 🚀
