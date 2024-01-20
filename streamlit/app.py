import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import warnings

# if  os.getcwd()=="C:\Users\metin":


# Suppress all warnings for better streamlit presentation
warnings.filterwarnings("ignore")

st.write("Current Directory:", os.getcwd())
st.write("Absolute Path of Data File:", os.path.abspath('data/client_data.csv'))

# data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
file_path1 = os.path.abspath('data/client_data.csv')
file_path2 = os.path.abspath('data/price_data.csv')
client_df = pd.read_csv(file_path1,index_col=0)
price_df = pd.read_csv(file_path2,index_col=0)

sns.set(style="whitegrid")

# Title of your page
st.title('Customer Retention Analysis for PowerCo')

# Introduction section
st.header('Introduction')
st.markdown("""
- **Dataset Origin**: Acquired from a virtual internship program offered by BCG on the Forage platform.
- **Client Concern**: PowerCo, a prominent gas and electricity utility company, is experiencing challenges in retaining customers.
- **Objective**: Perform exploratory data analysis, clean the data, and develop a predictive model to help PowerCo retain its customers.
- **Program Link**: [BCG Data Science Virtual Experience](https://www.theforage.com/simulations/bcg/data-science-ccdz)
""")

# Dataset Overview
st.header('Dataset Overview')
st.markdown("""
- **Client Data (`client_data`)**: 
  - Records: 14,606
  - Columns: 26
  - Primary Focus
- **Price Data (`price_data`)**: 
  - Records: 193,002
  - Columns: 8
  - Usage: To generate features for enhancing the `client_data` dataset
""")

if st.checkbox("Show client_data"):
    st.dataframe(client_df)
if st.checkbox("Show price_data"):
    st.write(price_df)
if st.checkbox("Show Data Descriptions"):
    st.markdown("""
### `client_data.csv`

- **id**: client company identifier
- **activity_new**: category of the company’s activity
- **channel_sales**: code of the sales channel
- **cons_12m**: electricity consumption of the past 12 months
- **cons_gas_12m**: gas consumption of the past 12 months
- **cons_last_month**: electricity consumption of the last month
- **date_activ**: date of activation of the contract
- **date_end**: registered date of the end of the contract
- **date_modif_prod**: date of the last modification of the product
- **date_renewal**: date of the next contract renewal
- **forecast_cons_12m**: forecasted electricity consumption for next 12 months
- **forecast_cons_year**: forecasted electricity consumption for the next calendar year
- **forecast_discount_energy**: forecasted value of current discount
- **forecast_meter_rent_12m**: forecasted bill of meter rental for the next 2 months
- **forecast_price_energy_off_peak**: forecasted energy price for 1st period (off peak)
- **forecast_price_energy_peak**: forecasted energy price for 2nd period (peak)
- **forecast_price_pow_off_peak**: forecasted power price for 1st period (off peak)
- **has_gas**: indicated if client is also a gas client
- **imp_cons**: current paid consumption
- **margin_gross_pow_ele**: gross margin on power subscription
- **margin_net_pow_ele**: net margin on power subscription
- **nb_prod_act**: number of active products and services
- **net_margin**: total net margin
- **num_years_antig**: antiquity of the client (in number of years)
- **origin_up**: code of the electricity campaign the customer first subscribed to
- **pow_max**: subscribed power
- **churn**: has the client churned over the next 3 months

### `price_data.csv`

- **id**: client company identifier
- **price_date**: reference date
- **price_off_peak_var**: price of energy for the 1st period (off peak)
- **price_peak_var**: price of energy for the 2nd period (peak)
- **price_mid_peak_var**: price of energy for the 3rd period (mid peak)
- **price_off_peak_fix**: price of power for the 1st period (off peak)
- **price_peak_fix**: price of power for the 2nd period (peak)
- **price_mid_peak_fix**: price of power for the 3rd period (mid peak)

**Note**: Some fields are hashed text strings. This preserves the privacy of the original data, but the commercial meaning is retained, and they may have predictive power.
""")
  
import streamlit as st

# Data Exploration Header
st.header('Data Exploration')

# Initial Observations
st.subheader('Initial Observations')
st.markdown("""
Even before delving into graphical analysis, we identified several key issues in the dataset:
1. **Handling Missing Values**: Targeting and addressing 'MISSING' values.
2. **Binary Conversion**: Transforming 't' and 'f' values to 1 and 0, respectively.
3. **Readable Labels**: Converting hashed text (like "lxidpiddsbxsbosboudacockeimpuepw") to more understandable labels.
4. **Date Conversion**: Changing date formats from object to datetime for improved visualization.
""")

# Imputation of Missing Values
st.subheader('Imputation of Missing Values')
st.markdown("""
Imputing missing values was primarily done using a Random Forest Classifier, focusing on numerical features to predict missing values. The imputation was mainly applied to 'channel_sales' and 'origin_up' features.
""")

# data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
file_path3 = os.path.abspath('data/cleaned_client_data.csv')
df = pd.read_csv(file_path3,index_col=0)


# Select numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64'])
numerical_columns_names = numerical_cols.columns.tolist()

# Slider for selecting column range
col_start, col_end = st.select_slider(
    'Select column range for correlation heatmap',
    options=numerical_columns_names,
    value=(numerical_columns_names[0], numerical_columns_names[-1])  # Default to full range
)

# Find the indices of the start and end columns in the list
start_idx = numerical_columns_names.index(col_start)
end_idx = numerical_columns_names.index(col_end) + 1

# Filter the DataFrame based on selected column range
filtered_df = numerical_cols[numerical_columns_names[start_idx:end_idx]]

# Compute the correlation matrix for the filtered DataFrame
corrMatrix = filtered_df.corr()

# Create the heatmap
fig, ax = plt.subplots(figsize=(16,12))
sns.heatmap(corrMatrix, annot=True, ax=ax)

# Display the heatmap in Streamlit
st.pyplot(fig)

st.markdown("""
We noticed that no single feature directly impacts churn significantly in numerical data. However, 'margin_gross_pow_ele' and 'margin_net_pow_ele' are perfectly correlated. This redundancy could potentially confuse the model. If accuracy issues arise, addressing these features could be a solution.
""")

figure_churn, ax = plt.subplots(figsize=(8, 6))
# Count the occurrences of each unique value in the 'churn' column
churn_counts = df['churn'].value_counts()
# Define colors for each bar
colors = ['#85C1E9', '#FF6B6B']  # 'blue' for Retention (0), 'red' for Churn (1)
# Create a bar plot with different colors
for i, index in enumerate(churn_counts.index):
    ax.bar(index, churn_counts[index], color=colors[i])
# Create a legend
ax.legend(["Retention", "Churn"])
# Adding labels and title
ax.set_xlabel('Churn Category')
ax.set_xticks([0, 1])
ax.set_ylabel('Counts')
ax.set_title('Churn Distribution')

# Churn Distribution
st.subheader('Churn Distribution')
st.pyplot(figure_churn)
st.markdown("""
- **Churn False Count**: 13,187
- **Churn True Count**: 1,419
- **Churn Ratio**: Approximately 9.7%

This indicates a slight data imbalance, where a naive model predicting 'False' for every instance would be about 90% accurate. Our focus, however, will be on enhancing the precision of true positives.
""")
# Insert churn distribution visual here

# Categorical Data Analysis
st.subheader('Categorical Data Analysis')
st.markdown("""
Analyzing categorical features for potential correlations with churn. For instance, different 'channel_sales' might indicate varying churn probabilities.
""")
category_feature = ['channel_sales','has_gas','origin_up']
binary_feature = 'churn' 

# Create a count plot
for category in category_feature:
    figure_categories,ax=plt.subplots(figsize=(8,6))
    sns.countplot(x=category, hue=binary_feature, data=df,ax=ax)
    ax.set_xlabel(category)
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of Categories with True/False Counts')
    st.pyplot(figure_categories)

# Customer Subscription Analysis
st.subheader('Customer Subscription Analysis')
# Convert 'start_date' to datetime and extract the year
df['date_activ'] = pd.to_datetime(df['date_activ']).dt.year
# Group by start year and churn, and count the occurrences
yearly_data = df.groupby(['date_activ', 'churn']).size().unstack(fill_value=0)
# Calculate ratios for retention (0) and churn (1) for each year
yearly_data_ratio = yearly_data.div(yearly_data.sum(axis=1), axis=0)
# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(12, 7))
yearly_data_ratio.plot(kind='bar', stacked=True, color=['#85C1E9', '#FF6B6B'], ax=ax)
# Annotate each bar with the percentage
for n, x in enumerate([*yearly_data_ratio.index.values]):
    for (proportion, y_loc) in zip(yearly_data_ratio.loc[x],
                                   yearly_data_ratio.loc[x].cumsum()):
        ax.text(x=n, y=(y_loc - proportion) + (proportion / 2), s=f'{proportion:.2%}', color='black', fontsize=10, ha='center')
# Adding labels and title
ax.set_xlabel('Start Year', fontsize=14)
ax.set_ylabel('Ratio', fontsize=14)
ax.set_title('Customer Retention and Churn Ratio by Start Year', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)  # Rotate x-axis labels for better readability
ax.tick_params(axis='y', labelsize=12)
# Adjust legend
ax.legend(title='Churn Status', fontsize=12, title_fontsize=14)
# Display the plot in Streamlit
st.pyplot(fig)
st.markdown("""
An interesting trend emerges when analyzing customer subscription duration. Both very new and very old customers show higher churn rates, with a relative stability observed in customers with 6-8 years of service. This suggests that long-term customer engagement strategies might be effective.
""")

st.markdown("""
Another visual for renewal year is made, closer the renewal year, higher the chance of churn.""")
# Convert 'start_date' to datetime and extract the year
df['date_renewal_year'] = pd.to_datetime(df['date_renewal']).dt.year
# Group by start year and churn, and count the occurrences
yearly_data = df.groupby(['date_renewal_year', 'churn']).size().unstack(fill_value=0)
# Calculate ratios for retention (0) and churn (1) for each year
yearly_data_ratio = yearly_data.div(yearly_data.sum(axis=1), axis=0)
# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(12, 7))
yearly_data_ratio.plot(kind='bar', stacked=True, color=['#85C1E9', '#FF6B6B'], ax=ax)
# Annotate each bar with the percentage
for n, x in enumerate([*yearly_data_ratio.index.values]):
    for (proportion, y_loc) in zip(yearly_data_ratio.loc[x],
                                   yearly_data_ratio.loc[x].cumsum()):
        ax.text(x=n, y=(y_loc - proportion) + (proportion / 2), s=f'{proportion:.2%}', color='black', fontsize=10, ha='center')
# Adding labels and title
ax.set_xlabel('Renewal Year', fontsize=14)
ax.set_ylabel('Ratio', fontsize=14)
ax.set_title('Customer Retention and Churn Ratio by Service Renewal Year', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)  # Rotate x-axis labels for better readability
ax.tick_params(axis='y', labelsize=12)
# Adjust legend
ax.legend(title='Churn Status', fontsize=12, title_fontsize=14)
# Display the plot in Streamlit
st.pyplot(fig)

# Merge the dataframes on client_id
merged_df = pd.merge(df, price_df, left_on='id', right_on='id')
# Merging Client and Price Data
st.subheader('Merging Client and Price Data')
# Plotting price trends
# Create the plot
figure_merge, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='price_date', y='price_off_peak_var', hue='churn', data=merged_df, ax=ax)
ax.set_title('Off-Peak Variable Price Trends Over Time by Churn Status')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12) 
# Set x-axis label
ax.set_xlabel('Months of 2015')

# Display the plot in Streamlit
st.pyplot(figure_merge)
st.markdown("""
We merged client information with price data to observe yearly pricing trends for customers and their correlation with churn.
""")


# Pricing Trends and Churn
st.markdown("""
Interestingly, churned customers had, on average, lower peak prices. This suggests that while pricing is a factor, it may not be the predominant reason for customer churn. However, the maximum price range for churned customers tends to be higher than for those who continued the service.
""")
# Insert pricing comparison visual here

# Feature Engineering Header
st.header('Feature Engineering')

# Introduction to Feature Engineering
st.markdown("""
In this step of virtual internship, we introduced a new dataset with additional price data for customers. Our goal is to enhance the original client dataset with relevant features extracted from this new information. We take columns that starts with "var" where new data is found. Then merge with our data.
""")
st.code("""
columns_starting_with_id_or_var = new_columns.filter(regex='^(id|var)')
client_df=pd.merge(client_df_old,columns_starting_with_id_or_var,on='id',how='left')
""", language='python')

# Date Related Features
st.subheader('New Features')
st.markdown("""
The features that I will implement:

**Date related**: 

From the dataset we can observe that dataset is in the perspective of 2016-01-29 (last date in date_modif_prod). We can create new features and find date differences to this date that explains date values better for model.
- date_end: We will find the difference between end of the contract and 2016-01-29.
- date_modif_prod: Find difference between 2016-01-29 and last date the contract was modified.

**Cost of Energy Consumption**:

We have the both the prices and consumption information. We can create a column that contains total cost of yearly consumption. As we don't have the monthly consumption information, we will find average price from yearly price data, to not pike the price to low levels because of zero values, we will only use non-zero valeus for mean.

The new created total_cost features makes the following assumption: Majority of consumption is made from electricity, thus electricity is selected as main factor in finding total cost. 

This is not fully true in physics sence as the units of energy and electricity does not meet. However <strong>purpose of the feature is to introduce a connection with electiricity consumption and average price to find a hypothetical yearly cost to customer.</strong>

            """)

# We want check numerical data, especially consumptions and forecasts:
consumption = client_df[[    'cons_12m', 
                            'cons_gas_12m', 
                            'cons_last_month',
                            'forecast_cons_12m', 
                            'forecast_cons_year', 
                            'forecast_discount_energy',
                            'forecast_meter_rent_12m', 
                            'forecast_price_energy_off_peak',
                            'forecast_price_energy_peak', 
                            'forecast_price_pow_off_peak',
                            'imp_cons']]
# Display Skewed Data Distribution Visual
st.subheader('Part 2 - Further Feature Engineering')
st.markdown("""
After moving on with previous version of dataset, model had low true positive accuracy. Thus taking a step back, we returned dataset, and a warning of virtual internship was on skewed data. So we take a look at numerical consumption and forecast data, which show us high skewed data:
            """)
fig,ax=plt.subplots(figsize=(12,6))
sns.histplot(data=consumption, x="cons_12m",kde=True,ax=ax)
st.pyplot(fig)

st.markdown('**The skew information for target features**')
for column in consumption.columns:
    st.write(column,": ",consumption[column].skew())

import numpy as np
# Apply log10 transformation, addition of plus + 1 for log function incase there are values that equal to 0
client_df["cons_12m"] = np.log10(client_df["cons_12m"] + 1)
client_df["cons_gas_12m"] = np.log10(client_df["cons_gas_12m"] + 1)
client_df["cons_last_month"] = np.log10(client_df["cons_last_month"] + 1)
client_df["forecast_cons_12m"] = np.log10(client_df["forecast_cons_12m"] + 1)
client_df["forecast_discount_energy "] = np.log10(client_df["forecast_discount_energy"] + 1)
client_df["forecast_cons_year"] = np.log10(client_df["forecast_cons_year"] + 1)
client_df["forecast_meter_rent_12m"] = np.log10(client_df["forecast_meter_rent_12m"] + 1)
client_df["forecast_price_pow_off_peak"] = np.log10(client_df["forecast_price_pow_off_peak"] + 1)
client_df["imp_cons"] = np.log10(client_df["imp_cons"] + 1)

# We want check numerical data, especially consumptions and forecasts:
consumption = client_df[[    'cons_12m', 
                            'cons_gas_12m', 
                            'cons_last_month',
                            'forecast_cons_12m', 
                            'forecast_cons_year', 
                            'forecast_discount_energy',
                            'forecast_meter_rent_12m', 
                            'forecast_price_energy_off_peak',
                            'forecast_price_energy_peak', 
                            'forecast_price_pow_off_peak',
                            'imp_cons']]

# Explanation of Skewness
st.markdown("""
Skewness indicates the asymmetry of the data distribution:
- **Moderate Skewness**: Between -1 and -0.5 (negative) or 0.5 and 1 (positive).
- **Substantial Skewness**: Less than -1 (negative) or greater than 1 (positive).

Our data showed significant skewness, which might have impacted the model's true positive accuracy.
""")

# Log Transformation and New Visual
st.subheader('Addressing Skewness with Log Transformation')
st.markdown("""
To address the skewness, we applied a log transformation to the skewed features:
""")
st.code('client_df["cons_12m"] = np.log10(client_df["cons_12m"] + 1)', language='python')
for column in consumption.columns:
    st.write(column,": ",consumption[column].skew())
st.markdown("This transformation resulted in a more normalized distribution:")
# Insert your plot for distribution after log transformation here
fig,ax=plt.subplots(figsize=(12,6))
sns.histplot(data=consumption, x="cons_12m",kde=True,ax=ax)
st.pyplot(fig)

st.markdown("""
With these transformations and feature enhancements, we're now ready to revisit the model training and evaluate any improvements in performance.
""")

# Models Section
st.header('Model Development')

# Introduction to Models Used
st.markdown("""
We explored two different models for our analysis:
- Random Forest Classifier
- Sequential Neural Network
""")

# Random Forest Classifier
st.subheader('Random Forest Classifier')


st.code("""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

X = df.drop(columns='churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight={0: 1, 1: 10}, n_estimators=250, bootstrap=False)
model.fit(X_train, y_train)
    """, language='python')

# Show the results
st.subheader('Random Forest Classifier Results')
st.text("""
Results:
              precision    recall  f1-score   support

           0       1.00      0.90      0.95      2893
           1       0.08      0.83      0.14        29

    accuracy                           0.90      2922
   macro avg       0.54      0.87      0.55      2922
weighted avg       0.99      0.90      0.94      2922
""")

st.markdown("""
As we can see accuracy seem to be high, but precision for true positive is %8 percent, which is extremely low.
            """)


# Sequential Neural Network
st.header('Sequential Neural Network')

# Data Scaling
st.subheader('Data Scaling')
st.markdown("""
Neural Networks require scaled data for efficient training. We use the Standard Scaler on our dataset:
""")
st.code("""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
""", language='python')

# Setting up GPU
st.subheader('Setting Up GPU for Computation')
st.markdown("""
To enhance training speed, we configure TensorFlow to use GPU:
""")
st.code("""
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
""", language='python')

# First Neural Network Model
st.subheader('First Neural Network Model')
st.markdown("""
Our initial model is a simple sequential neural network with three layers.
""")
# Code for the first model
st.code("""
model = Sequential([
    Dense(128, activation='relu', input_shape=(50,)), 
    Dense(32, activation='relu', input_shape=(50,)), 
    Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Fit the model with class weights
history = model.fit(X_train, y_train, batch_size=40, epochs=150, validation_split=0.2, verbose=1, class_weight=class_weight_dict)
    """, language='python')

# Results for the first model
st.subheader('Results for the First Model')
st.text("""
Results:
              precision    recall  f1-score   support
           0       0.86      0.91      0.89      2481
           1       0.28      0.19      0.23       441
    accuracy                           0.80      2922
   macro avg       0.57      0.55      0.56      2922
weighted avg       0.78      0.80      0.79      2922
""")

# Second Neural Network Model
st.subheader('Second Neural Network Model')
st.markdown("""
The second model is more complex, including additional layers and dropout to prevent overfitting. We also introduce model checkpoints and TensorBoard for monitoring.
""")
st.code("""
 Setup the ModelCheckpoint callback
checkpoint = ModelCheckpoint('../checkpoints/model_epoch_{epoch:02d}.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=False, 
                             save_weights_only=True, 
                             mode='auto', 
                             save_freq='epoch')

# Setup the TensorBoard callback
logdir = os.path.join("../logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)
    """, language='python')

# Code for the second model
st.code("""
model = Sequential([
    Dense(128, activation='relu', input_shape=(50,)), 
    Dense(512, activation='relu', input_shape=(50,)), 
    Dropout(0.1),
    Dense(512, activation='relu', input_shape=(50,)),
    Dropout(0.1), 
    Dense(512, activation='relu', input_shape=(50,)), 
    Dropout(0.1),
    Dense(256, activation='relu', input_shape=(50,)), 
    Dense(64, activation='relu', input_shape=(50,)),
    Dense(32, activation='relu', input_shape=(50,)), 
    Dropout(0.1),
    Dense(16, activation='relu', input_shape=(50,)), 
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])
    """, language='python')

# Results for the second model
st.subheader('Results for the Second Model')
st.text("""
Results:
              precision    recall  f1-score   support
           0       0.99      0.90      0.94      2854
           1       0.11      0.49      0.18        68
    accuracy                           0.89      2922
   macro avg       0.55      0.69      0.56      2922
weighted avg       0.97      0.89      0.93      2922
""")

st.markdown("""
Despite the more complex architecture, the true positive precision decreased. This suggests a need to revisit the data and perform further feature engineering. At this point we return to data and do feature engineering again, and as explained in feature engineering topic, we have second part. The second part exactly starts from here, thus we go back make changes, and return again for our next model:
""")

# Final Neural Network Model
st.header('Final Neural Network Model')

# Approach for the Final Model
st.subheader('Model Training Approach')
st.markdown("""
After addressing skewness in the data, we refined our model. We opted for a structure similar to the first model but increased the number of epochs to 500 for deeper training.
""")
# Code for the final model
st.code("""
model = Sequential([
    Dense(128, activation='relu', input_shape=(51,)), 
    Dropout(0.2),
    Dense(16, activation='relu'), 
    Dense(1, activation='sigmoid')
])
history = model.fit(X_train, y_train, 
                    batch_size=40, 
                    epochs=500, 
                    validation_split=0.2, 
                    verbose=1, 
                    class_weight=class_weight_dict,
                    callbacks=[checkpoint, tensorboard_callback])
    """, language='python')

# Results for the final model
st.subheader('Results for the Final Model')
st.text("""
Results:
              precision    recall  f1-score   support
           0       0.86      0.91      0.89      2467
           1       0.30      0.20      0.24       455
    accuracy                           0.80      2922
   macro avg       0.58      0.56      0.56      2922
weighted avg       0.77      0.80      0.79      2922
""")

# Result and Challenges
st.subheader('Result and Challenges')
st.markdown("""
**Achievements**:
- Achieved 30% true positive precision, significant for business value. It enables proactive measures to retain customers at risk of churn.

**Challenges**:
- The model's lower precision indicates a risk of false positives, potentially leading to increased customer support costs.
- Balancing precision and support costs is crucial. A lower threshold for churn prediction could increase precision but also elevate support costs.


**The model provides valuable insights for customer retention strategies. However, a balance must be struck between identifying at-risk customers and managing support resources efficiently.**
""")

# Conclusion
st.header('Conclusion')
st.markdown("""
While our model provides a significant tool for identifying potential churn, it's imperative to consider the operational and financial implications of its predictions. The model serves as a starting point for further refinement and operational strategy development.
""")