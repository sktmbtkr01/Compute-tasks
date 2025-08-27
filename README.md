# Compute-tasks
# 🥊 UFC Fighters - Exploratory Data Analysis (EDA)

This project explores a dataset of ~4,000 UFC fighters, covering their **physical attributes**, **fight records**, and **performance metrics**.  
The aim was to clean and prepare the data, handle missing values, engineer useful features, and uncover insights into how different attributes relate to fighter performance.

---

## 🔧 Workflow
1. **Data Cleaning**
   - Converted `DOB` to datetime and calculated fighter **Age**.  
   - Dropped fighters with **0 fights** to avoid unstable stats.  
   - Standardized numeric columns (height, reach, weight, percentages).  

2. **Feature Engineering**
   - Added `Fights = Wins + Losses + Draws`.  
   - Added `Win% = Wins / Fights * 100`.  
   - Derived metrics such as **Ape Index** (Reach − Height) and **BMI** (optional).  

3. **Handling Missing Values**
   - **Reach** imputed using **linear regression** with height (and weight when available).  
   - Height/Reach set to `0` only when no information could be recovered.  
   - Some categorical fields (e.g., stance, DOB) left as `NaN` if unavailable.  

4. **Exploratory Analysis**
   - Distribution of **Height, Reach, Weight, Win%**.  
   - **Height vs Reach** relationship + outliers.  
   - Stance frequency and performance comparisons.  
   - Striking metrics: Strikes/min, Accuracy, Absorbed/min, Defense.  
   - Grappling metrics: TDs/15, TD Acc %, TD Def %, Subs/15.  
   - Correlation heatmap for all numeric metrics.  
   - Radar (spider) charts to profile fighters across multiple skills.  

---

## 📊 Key Insights
- **Height vs Reach**: Strong positive linear correlation, with a few outliers.  
- **Experience matters**: Fighters with more bouts have more stable Win%.  
- **Stance**: Orthodox is dominant; stance alone doesn’t drive win rates.  
- **Balanced profiles** (good striking + grappling) stand out in radar charts.  

---

## 🛠️ Tools
- Python (Pandas, NumPy)  
- Matplotlib & Seaborn (EDA plots)  
- Plotly (interactive visuals)  
- Scikit-learn (linear regression for Reach imputation)  
- Jupyter Notebook

  ## 📊 Sample Visuals

### Height vs Reach
![Height vs Reach](images/height-vs-reach.png)

### Stance Distribution
![Stance Distribution](images/stance.png)

### Win% vs Fights
![Height-Reach correlation](images/correlation.png)



# 🥊 UFC Fighter Comparison App

A **Streamlit web app** to compare UFC fighters side-by-side across their physical stats, records, and performance metrics.  
Built using **Python, Pandas, Plotly, and Streamlit**.

🔗 **Live App:** [UFC Fighter Comparison](https://ufc-fighter-comparison-sktmbtkr.streamlit.app/)

---

## 🚀 Features

- Select **two fighters** from the UFC dataset.
- See **player cards** with:
  - Name, Nickname, Stance, Age
  - Height, Reach, Weight (with percentile badges)
  - Record: Wins–Losses–Draws, Fights, Win%
- **Attributes section** (Striking & Grappling metrics with percentiles).
- **Radar chart** comparing normalized performance across multiple metrics.
- Clean, responsive design built in Streamlit.

---

## 📊 Dataset

- Source: UFC fighters statistics dataset (`ufc_clean.csv`).
- Includes stats such as:
  - Wins, Losses, Draws, Win%
  - Strikes landed/min, Striking accuracy, Strikes absorbed/min
  - Takedowns per 15 min, Takedown accuracy, Takedown defense
  - Submissions attempted/15 min
  - Physical stats: Height, Reach, Weight

---

## 🛠️ Tech Stack

- [Python](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [Plotly](https://plotly.com/python/) (interactive charts)  
- [Streamlit](https://streamlit.io/) (web framework)  

---

  ## 📊 Sample Visuals

![](images/image2.png)

![](images/image1.png)

![](images/image3.png)
