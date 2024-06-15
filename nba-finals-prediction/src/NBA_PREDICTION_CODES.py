import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFECV
import seaborn as sns
import os

def clean_dataframe(df):
    """
    Cleans the DataFrame by removing empty columns and unnamed columns.
    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def load_and_clean_data(years, data_dir='~/Documents/nba-finals-prediction/data'):
    """
    Loads and cleans advanced and per 100 possession stats for given years.
    Parameters:
    years (range): The range of years to load the data for.
    data_dir (str): The directory where the CSV files are located.
    Returns:
    pd.DataFrame, pd.DataFrame: Cleaned advanced and per 100 possession stats DataFrames.
    """
    data_dir = os.path.expanduser(data_dir)
    advanced_stats = pd.DataFrame()
    per100_stats = pd.DataFrame()

    for year in years:
        adv_path = os.path.join(data_dir, f'advanced_stats_{year}.csv')
        per100_path = os.path.join(data_dir, f'per100_stats_{year}.csv')
        
        temp_adv = pd.read_csv(adv_path)
        temp_adv['Year'] = year
        temp_adv = clean_dataframe(temp_adv)
        advanced_stats = pd.concat([advanced_stats, temp_adv], ignore_index=True)
        
        temp_per100 = pd.read_csv(per100_path)
        temp_per100['Year'] = year
        temp_per100 = clean_dataframe(temp_per100)
        per100_stats = pd.concat([per100_stats, temp_per100], ignore_index=True)

    return advanced_stats, per100_stats



def prepare_final_dataset(advanced_stats, per100_stats, finals_data_filename, data_dir='~/Documents/nba-finals-prediction/data'):
    """
    Prepares the final dataset by merging advanced stats, per 100 possession stats, and finals data.
    Parameters:
    advanced_stats (pd.DataFrame): The advanced stats DataFrame.
    per100_stats (pd.DataFrame): The per 100 possession stats DataFrame.
    finals_data_filename (str): The filename of the finals data CSV file.
    data_dir (str): The directory where the CSV files are located.
    Returns:
    pd.DataFrame, pd.Series: The final dataset and the teams for 2024.
    """
    data_dir = os.path.expanduser(data_dir)
    finals_data_path = os.path.join(data_dir, finals_data_filename)
    finals_data = pd.read_csv(finals_data_path)
    finals_data = pd.melt(finals_data, id_vars=['Year', 'Win_team'],
                          value_vars=['East_team', 'West_team'], 
                          var_name='Conference', value_name='Team')
    finals_data['Is_Winner'] = (finals_data['Team'] == finals_data['Win_team']).astype(int)

    combined_stats = pd.merge(advanced_stats, per100_stats, on=['Year', 'Team'], 
                              suffixes=('_adv', '_per100'))
    final_dataset = pd.merge(combined_stats, finals_data, on=['Year', 'Team'])

    teams_2024 = finals_data[finals_data['Year'] == 2024]['Team']
    final_dataset = final_dataset.drop(['Rk', 'Team', 'Conference', 'Win_team', 'W', 'L', 'W/L%', 'Rk_adv'], axis=1)
    return final_dataset, teams_2024


def preprocess_data(final_dataset):
    """
    Preprocesses the data by handling missing values and splitting into features and target.
    Parameters:
    final_dataset (pd.DataFrame): The final dataset.
    Returns:
    pd.DataFrame, pd.Series: The preprocessed feature matrix and target vector.
    """
    X = final_dataset.drop(['Year', 'Is_Winner'], axis=1)
    y = final_dataset['Is_Winner']
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed, y

def train_model(X, y):
    """
    Trains a Random Forest model using RFECV for feature selection and evaluates it using cross-validation.
    Parameters:
    X (pd.DataFrame): The feature matrix.
    y (pd.Series): The target vector.
    Returns:
    object, list: The trained model and selected feature names.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFECV(model, step=1, cv=5, scoring='accuracy')
    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.support_]
    print(f"Selected features by Random Forest: {selected_features}")

    cv_scores = cross_val_score(model, X_train.iloc[:, selector.support_], y_train, cv=5)
    print(f"Cross-validated scores for Random Forest: {cv_scores}")
    print(f"Average CV score for Random Forest: {np.mean(cv_scores)}")

    model.fit(X_train.iloc[:, selector.support_], y_train)
    return model, selected_features

def visualize_feature_importances(model, selected_features):
    """
    Visualizes the feature importances of the trained model.
    Parameters:
    model (object): The trained model.
    selected_features (list): The selected feature names.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(selected_features)), importances[indices], align="center")
    plt.xticks(range(len(selected_features)), selected_features[indices], rotation=90)
    plt.xlim([-1, len(selected_features)])
    plt.show()

def predict_2024_winner(model, X_2024, teams_2024, imputer, selected_features):
    """
    Predicts the winner for the 2024 NBA finals using the trained model.
    Parameters:
    model (object): The trained model.
    X_2024 (pd.DataFrame): The 2024 feature matrix.
    teams_2024 (pd.Series): The teams in the 2024 finals.
    imputer (object): The imputer used for handling missing values.
    selected_features (list): The selected feature names.
    Returns:
    str, float: The predicted winner team and the probability.
    """
    X_2024_prepared = pd.DataFrame(imputer.transform(X_2024.drop(['Year', 'Is_Winner'], axis=1)),
                                   columns=X_2024.drop(['Year', 'Is_Winner'], axis=1).columns)
    X_2024_prepared = X_2024_prepared[selected_features]
    probabilities = model.predict_proba(X_2024_prepared)
    winner_index = np.argmax(probabilities[:, 1])
    final_predictions = np.zeros(probabilities.shape[0], dtype=int)
    final_predictions[winner_index] = 1
    predicted_winner_team = teams_2024.iloc[winner_index]
    predicted_winner_proba = probabilities[winner_index, 1]

    print(f"2024 Predictions by Random Forest: {final_predictions}")
    print(f"Probabilities of winning: {probabilities[:, 1]}")

    print(f"Predicted Winner for 2024: {predicted_winner_team} with a probability of {predicted_winner_proba:.2f}")
    return predicted_winner_team, predicted_winner_proba

def visualize_winning_probabilities(teams_2024, probabilities):
    """
    Visualizes the winning probabilities for the 2024 NBA finals.
    Parameters:
    teams_2024 (pd.Series): The teams in the 2024 finals.
    probabilities (np.ndarray): The winning probabilities for each team.
    """
    teams_2024 = teams_2024.reset_index(drop=True)  # Reset index to match prediction array
    plt.figure(figsize=(8, 6))
    plt.bar(teams_2024, probabilities[:, 1], color=['blue', 'green'])
    plt.xlabel('Teams')
    plt.ylabel('Probability of Winning')
    plt.title('Predicted Probability of Winning for 2024 NBA Finals')
    plt.show()

def visualize_correlation_heatmap(X_imputed):
    """
    Visualizes the correlation heatmap of the feature matrix.
    Parameters:
    X_imputed (pd.DataFrame): The feature matrix.
    """
    plt.figure(figsize=(20, 16))  # Increase the figure size
    correlation_matrix = X_imputed.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 7})
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

# Automated tests
def test_load_and_clean_data():
    years = range(2010, 2012)  # Small subset for testing
    advanced_stats, per100_stats = load_and_clean_data(years)
    assert not advanced_stats.empty, "Advanced stats should not be empty"
    assert not per100_stats.empty, "Per 100 stats should not be empty"
    print("Test load_and_clean_data passed")

def test_prepare_final_dataset():
    years = range(2010, 2012)  # Small subset for testing
    advanced_stats, per100_stats = load_and_clean_data(years)
    final_dataset, teams_2024 = prepare_final_dataset(advanced_stats, per100_stats, 'NBA_Finals_2010_2023.csv')
    assert not final_dataset.empty, "Final dataset should not be empty"
    assert not teams_2024.empty, "Teams 2024 should not be empty"
    print("Test prepare_final_dataset passed")

def test_preprocess_data():
    years = range(2010, 2012)  # Small subset for testing
    advanced_stats, per100_stats = load_and_clean_data(years)
    final_dataset, teams_2024 = prepare_final_dataset(advanced_stats, per100_stats, 'NBA_Finals_2010_2023.csv')
    X, y = preprocess_data(final_dataset)
    assert not X.empty, "Feature matrix should not be empty"
    assert not y.empty, "Target vector should not be empty"
    print("Test preprocess_data passed")

# Run tests
test_load_and_clean_data()
test_prepare_final_dataset()
test_preprocess_data()

# Load and clean data
years = range(2010, 2025)
advanced_stats, per100_stats = load_and_clean_data(years)

# Prepare final dataset
final_dataset, teams_2024 = prepare_final_dataset(advanced_stats, per100_stats, 'NBA_Finals_2010_2023.csv')

# Preprocess data
X, y = preprocess_data(final_dataset)

# Train model
model, selected_features = train_model(X, y)

# Predict 2024 winner
predicted_winner_team, predicted_winner_proba = predict_2024_winner(model,
    final_dataset[final_dataset['Year'] == 2024], 
    teams_2024, SimpleImputer(strategy='median').fit(X), selected_features)
