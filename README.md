    # Raw-Material (Metals & Minerals) Price Forecasting

## Project Title
**Raw-Material (Metals & Minerals) Price Forecasting**

## Business Problem
The business problem at hand is the unpredictable fluctuation in the prices of raw materials (Metals & Minerals), negatively impacting the cost structure and inventory management.

## Business Objectives
- Minimize procurement costs.
- Maximize profitability.
- Enhance competitiveness through efficient management of raw material (metals & minerals) sourcing and pricing strategies within the global market for mission-critical engineered solutions.

## Business Constraints
- Minimize the impact of price volatility on production costs.
- Optimize procurement strategies to ensure stable and affordable sourcing of raw materials (Metals & Minerals) for its engineered solutions.

## Business Success Criteria
- To optimize procurement strategies and reduce production costs by 10%.

## ML Success Criterion
- To achieve an accuracy of at least 95%.

## Economic Success Criteria
- To achieve cost savings in raw material procurement and inventory management of at least 20% (Based on the functionality discussion and current proceedings of procurement of raw materials).

## Project Summary
This project focuses on forecasting prices for eight metals using 348 records spanning 3-4 years. The following steps were conducted:

- Grouping and exploratory data analysis (EDA).
- Missing value checks, trend imputation, and outlier detection using the IQR method.
- Analysis of seasonality and trends.
- Training ten models per metal, evaluated with Mean Absolute Percentage Error (MAPE).
- Deployed Support Vector Regression (SVR) on Streamlit for local access.

## Table of Contents
1. [Data Collection](#data-collection)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Model Deployment](#model-deployment)
7. [Results and Findings](#results-and-findings)
8. [Conclusion and Future Work](#conclusion-and-future-work)
9. [References](#references)

---

### Data Collection
- Describe the data source and methodology used to gather the price data for metals and minerals.

### Data Preprocessing
- Discuss the steps taken to clean and prepare the data, including handling missing values and outlier detection.

### Exploratory Data Analysis (EDA)
- Provide insights into the techniques and visualizations used to explore the dataset.

### Feature Engineering
- Explain any new features created from the existing data to improve model performance.

### Model Training and Evaluation
- Overview of the models trained, including:
  - ARIMA
  - Random Forest
  - KNN
  - Simple Exponential Smoothing
  - XGBoost
  - LSTM
  - RNN
- Evaluation metrics used, focusing on MAPE (Mean Absolute Percentage Error).

### Model Deployment
- Description of how the SVR model was deployed using Streamlit for local access.

### Results and Findings
- Summary of the results obtained from the models and insights gained from the analysis.

### Conclusion and Future Work
- Key takeaways and potential improvements for future iterations of the project.

### References
- List of references or sources for the data and any external resources used.

---

## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

---

### MIT License

```
MIT License

Copyright (c) 2024 Sugumar S

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
