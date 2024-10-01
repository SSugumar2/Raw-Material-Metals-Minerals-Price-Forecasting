# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

# Set up the layout of the page
st.set_page_config(page_title="🔮 Raw-Material Price Forecasting", layout="wide")

# Title and introduction
st.title("🔮 Raw-Material Price Forecasting Dashboard")
st.markdown("""
    Welcome to the **Raw-Material Price Forecasting** application. Upload your data to forecast metal prices
    and get insightful results! 💡 The predictions are powered by KNN models to help optimize your procurement strategies
    and enhance profitability.
""")

# Sidebar for user input
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=['xlsx'])

# Processing uploaded data
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Perform mean imputation
    impute = SimpleImputer(strategy='mean')
    df['Price'] = df.groupby('Metals_Name')['Price'].transform(lambda x: impute.fit_transform(x.values.reshape(-1, 1)).ravel())

    st.sidebar.success("File uploaded successfully! 📂")
    
    st.write("## Data Preview:")
    st.dataframe(df.head())  # Show a preview of the uploaded data

    # User selects metals to forecast
    metals = df['Metals_Name'].unique()
    default_metals = metals[:8] if len(metals) >= 8 else metals
    selected_metals = st.multiselect(
        "Select Metals for Forecasting:",
        metals, 
        default=default_metals  # Default to first 8 metals or all available if fewer
    )

    if selected_metals:
        st.write(f"### Selected Metals: {', '.join(selected_metals)}")

        # Split the selected metals data for further analysis
        selected_df = df[df['Metals_Name'].isin(selected_metals)]

        # Showing summary statistics for selected metals
        st.write("## Summary Statistics 📊")
        summary_stats = selected_df.groupby('Metals_Name')['Price'].describe()
        st.dataframe(summary_stats)

        # Correlation matrix
        st.write("## Price Correlation Matrix 📈")
        pivot_df = selected_df.pivot(index='Month', columns='Metals_Name', values='Price')
        correlation_matrix = pivot_df.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

        # Outlier detection
        st.write("## Outlier Analysis 🚨")
        outliers = {}
        for metal, group in selected_df.groupby('Metals_Name'):
            q1 = group['Price'].quantile(0.25)
            q3 = group['Price'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier = group[(group['Price'] < lower_bound) | (group['Price'] > upper_bound)]
            outliers[metal] = outlier
        
        for metal, o in outliers.items():
            if not o.empty:
                st.warning(f"Outliers detected for {metal} 🔍")
                st.dataframe(o)
            else:
                st.success(f"No outliers detected for {metal} ✅")

        # Forecasting using pre-trained KNN models
        st.write("## Price Forecasting with KNN 🧠")
        forecast_results = {}

        for metal in selected_metals:
            try:
                with open(f'model_{metal}.pkl', 'rb') as model_file:
                    model = pickle.load(model_file)

                # Preparing test data for prediction
                test_x = pd.DataFrame({'Month': pd.to_datetime(pivot_df.index[-6:])})  # Assuming last 6 months as test set

                # Add lag features, checking for enough data
                for lag in [1, 2]:
                    lagged_values = pivot_df[metal].shift(lag).tail(6).values
                    test_x[f'{metal}_Lag{lag}'] = lagged_values
                
                # Drop rows with NaN, but check if any data is left
                test_x.dropna(inplace=True)

                if test_x.shape[0] == 0:
                    st.error(f"Not enough data available for {metal} after applying lag features. ❌")
                    continue

                test_y = pivot_df[metal].tail(6).values
                
                # Ensure test_y length matches test_x after dropping NaNs
                test_y = test_y[-len(test_x):]  

                # Generate predictions
                predictions = model.predict(test_x.drop(columns=['Month']))
                mape = mean_absolute_percentage_error(test_y, predictions) * 100

                forecast_results[metal] = {
                    'predictions': predictions,
                    'mape': mape
                }

                # Display results
                st.write(f"### Forecast for {metal} 📈")
                forecast_df = pd.DataFrame({
                    'Actual Price': test_y,
                    'Predicted Price': predictions
                }, index=pivot_df.index[-len(test_x):])

                st.line_chart(forecast_df)
                st.success(f"MAPE for {metal}: {mape:.2f}% 🎯")

            except FileNotFoundError:
                st.error(f"Model for {metal} not found! Please check the model file. ❌")
            except Exception as e:
                st.error(f"An error occurred for {metal}: {e}")

else:
    st.sidebar.info("Please upload a data file to proceed. 📄")
