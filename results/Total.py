import pandas as pd

df1 = pd.read_csv('D:\\Python Project\\Python Learn\\5261\\Emotion Data\\non_zero_impact_score_examples.csv')

df1['date'] = pd.to_datetime(df1['date'])
df1 = df1[(df1['date'] >= '2008-07-01') & (df1['date'] <= '2020-12-30')]

df1['impact_score'] = df1['impact_score'].astype(str).str.upper()

df1 = df1.groupby('date')['impact_score'].apply(list).reset_index()
df1.columns = ['Date', 'impact_score']

df2 = pd.read_csv('D:\\Python Project\\Python Learn\\5261\\New Data\\Cleaned Data\\Processed_Data.csv')
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2[(df2['Date'] >= '2008-07-01') & (df2['Date'] <= '2020-12-30')]


merged_df = pd.merge(df2, df1, on='Date', how='left')

merged_df['impact_score'] = merged_df['impact_score'].apply(lambda x: x if isinstance(x, list) else [0])

merged_df.to_csv('D:\\Python Project\\Python Learn\\5261\\New Code\\merged_data.csv', index=False)

# DevelopTime: 4/29/2024 10:39 PM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

path_processed = r'D:\Python Project\Python Learn\5261\New Code\Other Exchange\Database2.csv'
data = pd.read_csv(path_processed, parse_dates=['Date'])
data.dropna(inplace=True)

data.reset_index(drop=True, inplace=True)

interest_columns = [
    'impact_score', 'SGD_GDP_Billions', 'USD_GDP_Billions',
    'Singapore_Inflation', 'USA_Inflation', 'SG_Interest_Rate',
    'US_Interest_Rate', 'Price', 'STI', 'ExchangeRate',
    'Daily Exports(millions)', 'Daily Imports(millions)',
    'Daily Balance(millions)', 'FOREIGN RESERVES (US$ MILLION)',
    'GoldPrice', 'DXI', 'USD_EUR_ExchangeRate', 'USD_JPY_ExchangeRate',
    'USD_CNY_ExchangeRate'
]



for col in interest_columns:  # Now includes 'SGD_GDP_Billions'
    data[f'{col}_return'] = data[col].pct_change()

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data.dropna(inplace=True)

fig, axes = plt.subplots(nrows=len(interest_columns)-1, ncols=1, figsize=(10, (len(interest_columns)-1)*5))
for i, col in enumerate(interest_columns[1:]):  # Exclude 'Date' from visualization
    sns.lineplot(data=data, x='Date', y=f'{col}_return', ax=axes[i], label=f'{col} Return')
    axes[i].set_title(f'{col} Return Over Time')
plt.tight_layout()
plt.show()

print('Data Statistics:\n', data.describe())

corr_matrix = data[[f'{col}_return' for col in interest_columns[1:]]].corr()
print('Correlation Matrix:\n', corr_matrix)


plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Returns')
plt.show()

high_corr_threshold = 0.7
for col in corr_matrix.columns:
    highly_correlated = corr_matrix.index[(corr_matrix[col] > high_corr_threshold) & (corr_matrix.index != col)].tolist()
    if highly_correlated:
        print(f"{col} has high correlation with: {', '.join(highly_correlated)}")


from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data[[f'{col}_return' for col in interest_columns if col != 'ExchangeRate']]
y = data['ExchangeRate_return']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)  # 保留95%的方差
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f'PCA selected {pca.n_components_} components')

# LassoCV
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_
print(f'Lasso selected features: {dict(zip(X.columns, lasso_coef))}')

# RandomForest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_feature_importance = rf.feature_importances_
print(f'Random Forest feature importance: {dict(zip(X.columns, rf_feature_importance))}')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

file_path = r'D:\Python Project\Python Learn\5261\New Code\Other Exchange\Database2.csv'
data = pd.read_csv(file_path)
data.dropna(inplace=True)
data['ExchangeRate_Short_MA'] = data['ExchangeRate'].rolling(window=20).mean()
data['ExchangeRate_Long_MA'] = data['ExchangeRate'].rolling(window=80).mean()

index_info = ['impact_score', 'SGD_GDP_Billions', 'USD_GDP_Billions',
              'Singapore_Inflation', 'USA_Inflation', 'SG_Interest_Rate',
              'US_Interest_Rate', 'Price', 'STI', 'ExchangeRate',
              'Daily Exports(millions)', 'Daily Imports(millions)',
              'Daily Balance(millions)', 'FOREIGN RESERVES (US$ MILLION)',
              'GoldPrice', 'DXI', 'USD_EUR_ExchangeRate', 'USD_JPY_ExchangeRate',
              'USD_CNY_ExchangeRate', 'ExchangeRate_Long_MA', 'ExchangeRate_Short_MA']

columns_to_calculate = index_info[1:]

for col in columns_to_calculate:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if (data[col] == 0).any():
            data[col] = data[col].replace(0, 0.001)
        data[f'{col}_return'] = data[col].pct_change()

data.dropna(inplace=True)

features = ['Price', 'STI', 'GoldPrice', 'DXI', 'USD_EUR_ExchangeRate', 'USD_CNY_ExchangeRate']

X = data[features]
y = data['ExchangeRate']

rf_model = RandomForestRegressor(n_estimators=500, random_state=42)

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')

print("MSE:", -cv_scores)
print("averge_MSE:", -cv_scores.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"RF (MSE): {mse_rf:.4f}")
print(f"RF (R²): {r2_rf:.4f}")

importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

forest_importances = pd.Series(importances, index=features)
fig, ax = plt.subplots(figsize=(6, 20))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, c='blue')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title('Random Forest Predicted vs Actual')
plt.show()

result = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
perm_importances = pd.DataFrame(result.importances_mean, index=features, columns=['Importance']).sort_values('Importance', ascending=False)
print("VIF：")
print(perm_importances)

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = 'D:/Python Project/Python Learn/5261 project/selected_data.parquet'
data = pd.read_parquet(data_path)

# Define keywords and their context for more accurate sentiment impact analysis
'''
需要修改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！加入金融术语
keywords_context = {
    'USD': {
        'keywords': [
            'Federal Reserve', 'interest rate hike', 'US inflation data', 'US GDP growth',
            'US unemployment rate', 'US trade balance', 'Federal budget deficit', 'US monetary policy'
        ],
        'positive': [
            'hike', 'strong', 'growth', 'surplus', 'tighten', 'rally'
        ],
        'negative': [
            'cut', 'weak', 'decline', 'deficit', 'loosen', 'slump'
        ]
    },
    'SGD': {
        'keywords': [
            'Monetary Authority of Singapore', 'SGD interest rates', 'Singapore GDP growth',
            'Singapore inflation rate', 'Singapore trade data', 'Singapore government budget'
        ],
        'positive': [
            'raise', 'strong', 'growth', 'surplus', 'tighten', 'advance'
        ],
        'negative': [
            'cut', 'weak', 'decline', 'deficit', 'loosen', 'retract'
        ]
    }
}
'''

def contains_keywords(text, keywords, positive, negative):
    text_lower = text.lower()
    keyword_hits = any(word in text_lower for word in keywords)
    if not keyword_hits:
        return 0  # No keywords found, skip processing
    pos_count = sum(text_lower.count(pos) for pos in positive)
    neg_count = sum(text_lower.count(neg) for neg in negative)
    return pos_count - neg_count


for key, context in keywords_context.items():
    data[f'{key}_context'] = data['text'].apply(
        lambda x: contains_keywords(x, context['keywords'], context['positive'], context['negative'])
    )


for key, context in keywords_context.items():
    data[f'{key}_context'] = data['text'].apply(lambda x: contains_keywords(x, context['keywords'], context['positive'], context['negative']))

analyzer = SentimentIntensityAnalyzer()
data['sentiment'] = data['short_description'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

data['impact_currency'] = 'Neutral'
data['impact_score'] = 0.0

def sentiment_impact(row):
    for currency in ['USD', 'SGD']:
        context_score = row[f'{currency}_context']
        if context_score != 0:
            adjusted_score = row['sentiment'] * context_score
            if currency == 'SGD':
                adjusted_score *= -1
            if abs(adjusted_score) > abs(row['impact_score']):  # Only update if the new score is more significant
                row['impact_currency'] = currency
                row['impact_score'] = min(1, max(-1, adjusted_score))
    return row

data = data.apply(sentiment_impact, axis=1)

average_sentiment = data.groupby('impact_currency')['impact_score'].mean()
print(average_sentiment)

print(data.head())

impact_score_1 = data[data['impact_score'] <= 1].head(5)
impact_score_minus_1 = data[data['impact_score'] >= -1].head(5)

print("Examples with impact_score = 1:")
print(impact_score_1[['short_description', 'impact_currency', 'impact_score']])

print("\nExamples with impact_score = -1:")
print(impact_score_minus_1[['short_description', 'impact_currency', 'impact_score']])


non_zero_impact_scores = data[data['sentiment'] != 0].head(1000)

csv_output_path = 'D:\\Python Project\\Python Learn\\5261\\Emotion Data\\non_zero_impact_score_examples.csv'

non_zero_impact_scores.to_csv(csv_output_path, index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import matplotlib.dates as mdates
from tensorflow.keras.layers import Attention
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Attention, Bidirectional, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv(r'D:\Python Project\Python Learn\5261\New Code\Other Exchange\Database2.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

data['ExchangeRate_Short_MA'] = data['ExchangeRate'].rolling(window=20, min_periods=1).mean()
data['ExchangeRate_Long_MA'] = data['ExchangeRate'].rolling(window=80, min_periods=1).mean()

data.dropna(inplace=True)

selected_features = [
    'Price', 'STI', 'GoldPrice', 'DXI',
    'USD_EUR_ExchangeRate', 'USD_CNY_ExchangeRate',
    'ExchangeRate_Short_MA', 'ExchangeRate_Long_MA',
    'impact_score'
]
features = data[selected_features]
target = data['ExchangeRate']

features.dropna(inplace=True)
target.dropna(inplace=True)


scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)
scaler_target = MinMaxScaler()
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(features_scaled, target_scaled.ravel())

rf_predictions = rf_model.predict(features_scaled)


rf_predictions_reshaped = rf_predictions.reshape(-1, 1)


extended_features = np.hstack((features_scaled, rf_predictions_reshaped))
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


import numpy as np


def plot_predictions(real_vals, lstm_predicted_vals, title="Predictions vs. Actual", num_dates=10):
    plt.figure(figsize=(14, 7))


    dates = data['Date'].iloc[train_size:train_size + len(real_vals)].map(mdates.date2num)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


    baseline_predicted_vals = np.roll(real_vals, 3)
    baseline_predicted_vals[:3] = real_vals[:3]

    plt.plot(dates, real_vals, label='Actual Exchange Rate', color='blue')
    plt.plot(dates, lstm_predicted_vals, label='LSTM Predicted Exchange Rate', color='red')
    plt.plot(dates, baseline_predicted_vals, label='Baseline Predicted Exchange Rate', color='green')

    random_indices = random.sample(range(len(real_vals)), num_dates)
    for idx in random_indices:
        date = data.iloc[train_size + idx]['Date']
        plt.scatter(mdates.date2num(date), real_vals[idx], color='blue')
        plt.scatter(mdates.date2num(date), lstm_predicted_vals[idx], color='red')
        plt.scatter(mdates.date2num(date), baseline_predicted_vals[idx], color='green')
    print(
        f"Date: {date.strftime('%Y-%m-%d')}, Actual: {real_vals[idx]}, LSTM Predicted: {lstm_predicted_vals[idx]}, Baseline Predicted: {baseline_predicted_vals[idx]}")

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


    lstm_mae = np.mean(np.abs(real_vals - lstm_predicted_vals))
    baseline_mae = np.mean(np.abs(real_vals - baseline_predicted_vals))

    print(f"Average MAE between LSTM predictions and actual values: {lstm_mae}")
    print(f"Average MAE between baseline predictions and actual values: {baseline_mae}")

    return lstm_mae, baseline_mae


time_steps_list = [3]
performance_metrics = {}

for time_steps in time_steps_list:
    X, y = create_dataset(features_scaled, target_scaled.ravel(), time_steps=time_steps)

    train_size = int(len(X) * 0.96)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(0.01)))(input_layer)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(0.01)))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(0.01)))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    query = x
    value = x
    attention = Attention(use_scale=True)([query, value])
    attention = Dropout(0.05)(attention)
    attention = BatchNormalization()(attention)

    x = Bidirectional(LSTM(200, kernel_regularizer=l2(0.01)))(attention)
    x = Dropout(0.3)(x)

    output = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=256,
                        validation_data=(X_val, y_val), verbose=1,
                        callbacks=[early_stop]
                        )

    predicted_val = model.predict(X_val)
    predicted_val_rescaled = scaler_target.inverse_transform(predicted_val)
    real_val_rescaled = scaler_target.inverse_transform(y_val.reshape(-1, 1))

    mse_val = mean_squared_error(real_val_rescaled, predicted_val_rescaled)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(real_val_rescaled, predicted_val_rescaled)
    r2_val = r2_score(real_val_rescaled, predicted_val_rescaled)

    performance_metrics[time_steps] = {
        'MSE': mse_val,
        'RMSE': rmse_val,
        'MAE': mae_val,
        'R^2': r2_val
    }



    plot_predictions(real_val_rescaled, predicted_val_rescaled,
                     title=f"LSTM Model Predictions for time_steps = {time_steps}")

for time_steps, metrics in performance_metrics.items():
    print(f"Results for time_steps = {time_steps}:")
    print(f"MSE: {metrics['MSE']}, RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, R^2: {metrics['R^2']}\n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import logging

def initialize_particles(num_particles, lower_bound, upper_bound):
    particles = np.random.uniform(low=-0.008, high=0.008, size=num_particles)
    weights = np.ones(num_particles) / num_particles
    return particles, weights


def predict_particles(particles, lstm_predict_function, features):
    predictions = lstm_predict_function(features)
    predicted_particles = predictions.flatten() + np.random.normal(0, 0.001, size=particles.shape)
    return predicted_particles


def update_particle_weights(particles, weights, actual_value, beta=0.1):
    particle_differences = np.abs(particles - actual_value)
    logging.debug(f"Before update: max weight={np.max(weights)}, min weight={np.min(weights)}")
    weights *= np.exp(-beta * particle_differences)
    weights += 1.e-30000
    weights /= np.sum(weights)
    logging.debug(f"After update: max weight={np.max(weights)}, min weight={np.min(weights)}")
    return weights

def stratified_resample(particles, weights):
    N = len(particles)
    positions = (np.random.rand(N) + range(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes], np.ones_like(weights) / N


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :]
        Xs.append(v)
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)


def train_lstm_model(X_train, y_train, X_val, y_val, time_steps):
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(200, return_sequences=True, kernel_regularizer=l2(0.01))(input_layer)
    x = Dropout(0.05)(x)
    x = BatchNormalization()(x)
    attention = Attention(use_scale=True)([x, x])
    attention = Dropout(0.05)(attention)
    attention = BatchNormalization()(attention)
    x = LSTM(200, kernel_regularizer=l2(0.01))(attention)
    x = Dropout(0.05)(x)
    output = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=[early_stop])
    return model

def lstm_predict_function(model, scaler, features):
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return scaler.inverse_transform(prediction).flatten()


def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predicted_vals = scaler.inverse_transform(predictions)

    actual_vals = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(actual_vals, label='Actual Values')
    plt.plot(predicted_vals, label='Predicted Values', alpha=0.7)
    plt.title('LSTM Model Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    mse = mean_squared_error(actual_vals, predicted_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_vals, predicted_vals)
    r2_val = r2_score(actual_vals, predicted_vals)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2:{r2_val}")


def plot_particle_trajectories(data, real_vals, particle_trajectories, train_size):
    with plt.style.context('ggplot'):
        plt.figure(figsize=(14, 7))
    valid_indices = ~np.isnan(real_vals).flatten()
    dates = data.iloc[train_size:train_size + len(real_vals)]['Date'][valid_indices]
    real_vals = real_vals[valid_indices]

    dates_mapped = dates.map(mdates.date2num)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.plot(dates_mapped, real_vals, label='Actual Log Returns', color='blue', alpha=0.75)

    for particles_at_time_t in particle_trajectories.T:
        plt.scatter(dates_mapped, particles_at_time_t, color='red', alpha=0.05)

    plt.title("Particle Filter Predictions over Time")
    plt.xlabel('Time')
    plt.ylabel('Log Returns')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


def main():
    data = pd.read_csv(r'D:\Python Project\Python Learn\5261\New Code\Other Exchange\Database2.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data['Log_Returns'] = np.log(data['ExchangeRate'] / data['ExchangeRate'].shift(1))
    data['Price_Log_Return'] = np.log(data['Price'] / data['Price'].shift(1))
    data['STI_Log_Return'] = np.log(data['STI'] / data['STI'].shift(1))
    data['GoldPrice_Log_Return'] = np.log(data['GoldPrice'] / data['GoldPrice'].shift(1))
    data['USD_EUR_ExchangeRate_Log_Return'] = np.log(
    data['USD_EUR_ExchangeRate'] / data['USD_EUR_ExchangeRate'].shift(1))
    data['USD_CNY_ExchangeRate_Log_Return'] = np.log(
    data['USD_CNY_ExchangeRate'] / data['USD_CNY_ExchangeRate'].shift(1))
    data['impact_score_Log_Return'] = np.log(data['impact_score'] / data['impact_score'].shift(1))
    data['ExchangeRate_Short_MA'] = data['ExchangeRate'].rolling(window=20, min_periods=1).mean()
    data['ExchangeRate_Long_MA'] = data['ExchangeRate'].rolling(window=80, min_periods=1).mean()
    data['ExchangeRate_Long_MA_Log_Return'] = np.log(
    data['ExchangeRate_Long_MA'] / data['ExchangeRate_Long_MA'].shift(1))
    data['ExchangeRate_Short_MA_Log_Return'] = np.log(
    data['ExchangeRate_Short_MA'] / data['ExchangeRate_Short_MA'].shift(1))
    selected_features = [
        'Price_Log_Return', 'STI_Log_Return', 'GoldPrice_Log_Return',
        'USD_EUR_ExchangeRate_Log_Return', 'USD_CNY_ExchangeRate_Log_Return',
        'ExchangeRate_Long_MA_Log_Return', 'ExchangeRate_Short_MA_Log_Return',
        'impact_score_Log_Return'
        ]
    features = data[selected_features]
    target = data['Log_Returns']
    features = data[selected_features].dropna()
    target = data['Log_Returns'].dropna()
    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)
    scaler_target = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))
    X, y = create_dataset(features_scaled, target_scaled.ravel(), time_steps=3)
    train_size = int(len(X) * 0.96)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    model = train_lstm_model(X_train, y_train, X_val, y_val, 3)
    predicted_val = model.predict(X_val)
    predicted_val_rescaled = scaler_target.inverse_transform(predicted_val)
    real_val_rescaled = scaler_target.inverse_transform(y_val.reshape(-1, 1))

    num_particles = 1000
    particles, weights = initialize_particles(num_particles, lower_bound=-0.05, upper_bound=0.05)
    particle_trajectories = np.zeros((len(X_val), num_particles))

    for i in range(len(X_val)):
        current_features = X_val[i]
        actual_value = y_val[i]

        particles = predict_particles(particles, lambda features: lstm_predict_function(model, scaler_target, features),
                                      current_features)
        weights = update_particle_weights(particles, weights, actual_value)
        particles, weights = stratified_resample(particles, weights)
        particle_trajectories[i] = particles

    plot_particle_trajectories(data, real_val_rescaled, particle_trajectories, train_size)
    evaluate_model(model, X_val, y_val, scaler_target)


if __name__ == "__main__":
    main()


