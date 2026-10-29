# Presentation, Demo & Report Guide

## A. Step-by-step workflow (do in this order)
1. `python generate_dataset.py` -> note the output shape (3530 x 9).
2. `python train.py` -> read the console output top to bottom. It prints: missing values, duplicates, clean shape, train/test sizes, model table, best model, per-class report, Linear Regression MAE/RMSE/R2.
3. Open `figures/` - every PNG is a figure for the report.
4. `streamlit run app.py` -> test 3 inputs (see demo script).
5. Push to GitHub -> deploy on Streamlit Cloud -> paste URL in report.

## B. How each mark-component is covered (what to say in the viva)

### 1. Data Cleaning & Splitting (3 marks)
| Step | What the code does | Why |
|---|---|---|
| Duplicates | `drop_duplicates()` removed about 146 rows | repeated rows bias the model and leak into the test set |
| Missing values | Delay_Minutes -> median; Weather -> mode | median is robust to outliers; mode keeps the most common category |
| Outliers | IQR rule, values capped (8 sensor-glitch delays such as 300 min) | impossible values distort scaling and KNN distances |
| Feature engineering | Hour_sin/cos, Time_Slot (Night, Morning/Evening Peak...), Is_Weekend | traffic repeats daily, so 23:00 and 00:00 must be 'close' |
| Encoding | One-Hot for Day/Weather/Area/Time_Slot/Holiday/Event/Previous_Traffic | models need numbers; one-hot avoids fake ordering (Monday < Tuesday) |
| Scaling | StandardScaler on Hour, Hour_sin, Hour_cos (for LR, KNN, NB) | distance/gradient models are scale-sensitive; trees are not |
| Split | 80/20, `stratify=y`, `random_state=42` | stratify keeps Low/Medium/High proportions equal; seed makes results reproducible |
| Pipeline | preprocessing inside `Pipeline` | prevents data leakage (scaler learns only from training data) |

Show: `01_outliers.png`, `02_class_distribution.png`.

### 2. Model Comparison & Performance Analytics (4 marks)
Models: Logistic Regression, Decision Tree, KNN, Random Forest, Naive Bayes (classification) + Linear Regression (delay minutes).
Metrics: Accuracy, weighted Precision, Recall, F1, 5-fold CV accuracy; MAE/RMSE/R2 for regression.

Sample result from this run (yours may differ - **use your own numbers from train.py**):
| Model | Acc | Prec | Rec | F1 | CV |
|---|---|---|---|---|---|
| Logistic Regression | 0.746 | 0.746 | 0.746 | 0.746 | 0.746 |
| Random Forest | 0.743 | 0.745 | 0.743 | 0.744 | 0.748 |
| Decision Tree | 0.740 | 0.738 | 0.740 | 0.739 | 0.723 |
| KNN | 0.713 | 0.711 | 0.713 | 0.712 | 0.739 |
| Naive Bayes | 0.691 | 0.682 | 0.691 | 0.665 | 0.672 |
Linear Regression (delay minutes): MAE ~1.9 min, RMSE ~2.5 min, R2 ~0.83.
Accuracy is ~75% because we removed the vehicle count (the strongest but impractical input). That is a realistic, honest result.

How to explain: The top models are within 1% of each other, so say 'Logistic Regression and Random Forest perform similarly; we select by F1-score'. Tree models capture non-linear rush-hour patterns. Naive Bayes is weakest because it assumes features are independent, but Hour, Time_Slot and Previous_Traffic are strongly related. Medium is the hardest class (see confusion matrix) because it sits between Low and High. Decision Tree is nearly as good and fully explainable (`08_decision_tree.png`). Show `05_model_comparison.png`, `06_confusion_matrices.png`, `07_feature_importance.png`, `09_linear_regression.png`.

### 3. UI Deployment & Cross-Platform Accessibility (4 marks)
Streamlit app, deployed on Streamlit Community Cloud. Runs in any browser: Windows, Linux, macOS, Android, iOS - no installation for the user. Layout is responsive (columns collapse on mobile). Show the live URL on your phone during the demo.

### 4. Code Quality, GitHub, Report (4 marks)
Commented code, a clear file structure, README, requirements.txt, regular commits (commit after each step, not once at the end), the report as PDF in the repo.

## C. Live demo script (5 minutes)
1. (30 s) Problem: congestion is reactive; we predict it in advance.
2. (60 s) Run `train.py`; point at missing values -> clean shape -> train/test sizes.
3. (60 s) Show comparison table + bar chart; name the best model and justify.
4. (90 s) Open the web app. Try:
   - Live mode: show it filling time, day and weather automatically, pick an area, and watch the prediction update.
   - Manual: Hour 8, Monday, Downtown, Clear, previous High -> **High** 🔴
   - Manual: Hour 2, Sunday, Residential, Clear, previous Low -> **Low** 🟢
   - Manual: Hour 13, Wednesday, Industrial, Cloudy, previous Medium -> **Medium** 🟡
   - Scroll to the 24-hour forecast and read out the 'High traffic expected at' hours.
5. (30 s) Show the live link on your phone + GitHub repo.
6. (30 s) Future work: real-time sensor/Google Maps data, LSTM time-series models, parking/pollution modules.

## D. Likely viva questions
- *Why stratify?* Keeps class ratio the same in train and test.
- *Why pipelines?* Avoid data leakage; same preprocessing at prediction time.
- *Accuracy vs F1?* F1 balances precision and recall; better when classes are uneven.
- *Why Linear Regression isn't in the accuracy table?* It predicts a number, so we use MAE/RMSE/R2.
- *Overfitting?* Check train vs test and 5-fold CV; depth limited on the Decision Tree.
- *Is the data real?* Be honest: the default dataset is synthetic (simulated rush hours). Swap in a Kaggle traffic CSV to use real data (change column names at top of train.py) - the pipeline is unchanged.
- *What is Find-S?* Finds the most specific hypothesis covering all positive examples; ignores negatives; fails on noisy data - so we use it only as a concept demo.
- *Why Random Forest over Decision Tree?* Ensemble reduces variance; slightly higher CV score.

## E. 10-page report outline
| Section | Pages | Content |
|---|---|---|
| Title, abstract, contents | 0.5 | |
| 1. Introduction | 1 | problem, motivation, objectives, smart-city context |
| 2. Literature Review | 1.5 | 5+ recent papers in a comparison table |
| 3. Methodology | 1 | pipeline flow diagram, tools, dataset description |
| 4. Data cleaning & preprocessing | 1.5 | table from section B1, outlier figure, class distribution |
| 5. Models comparison (>=4) | 2 | comparison table, bar chart, confusion matrices, feature importance, regression results |
| 6. Results & discussion | 1 | best model, why, limitations (synthetic data, no live feed) |
| 7. UI development & deployment | 1 | screenshots (desktop + mobile), Streamlit URL, deployment steps |
| 8. Conclusion & future work | 0.5 | |
| References | 0.5 | |

### Literature review - how to do it (important)
Find **5 papers from 2020-2025** on Google Scholar / IEEE Xplore / MDPI / ScienceDirect. Search terms:
`traffic congestion prediction machine learning`, `short-term traffic flow forecasting LSTM`, `traffic prediction random forest XGBoost`, `smart city traffic prediction deep learning`, `graph neural network traffic forecasting`.
Read the abstract + conclusion of each and fill this table (do **not** copy any paper's text or invent citations - examiners check):

| Ref | Authors, Year | Dataset | Methods | Best result | Limitation | How ours differs |
|---|---|---|---|---|---|---|

Then write 1 paragraph comparing them (e.g. "Deep-learning models such as LSTM [2] achieve higher accuracy but need large datasets, whereas our tree-based approach is lightweight and interpretable...").

## F. Extra bits that impress
- Add `Previous_Traffic` ablation: retrain without it and compare accuracy.
- Run `GridSearchCV` on the Decision Tree `max_depth`.
- Add XGBoost as a 6th model.
