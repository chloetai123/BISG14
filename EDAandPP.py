# EDAandPP.py

# Import the library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import joblib
import sys

# Make sure plots don't open as windows
plt.switch_backend("Agg")

sns.set_theme(style="whitegrid")

FIG_DIR = "eda_outputs"
os.makedirs(FIG_DIR, exist_ok=True)
FIG_COUNT = 1  # simple counter to avoid overwriting

# Redirect ALL terminal output (print) into one file
terminal_path = os.path.join(FIG_DIR, "terminal_output.txt")
orig_stdout = sys.stdout
sys.stdout = open(terminal_path, "w", encoding="utf-8")

try:
    # Define the file path to your dataset (replace 'your_dataset.csv' with the actual file path)
    file_path = 'loan_data.csv'

    # Use Pandas to read the CSV file into a DataFrame
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(file_path)

        # If successful, display the first few rows of the dataset to verify
        print("Data successfully loaded. Here are the first few rows:")
        print(df.head())

        # Display basic information about the dataset
        print("\nDataset Info:")
        print(df.info())

        # Display summary statistics for numerical columns
        print("\nSummary Statistics:")
        print(df.describe())

    except FileNotFoundError:
        # Handle the case where the file is not found
        print("File not found. Please check the file path.")
        raise
    except Exception as e:
        # Handle other potential exceptions
        print(f"An error occurred: {str(e)}")
        raise

    # Convert 'credit.policy' and 'not.fully.paid' columns to boolean type
    df['credit.policy'] = df['credit.policy'].astype(bool)
    df['not.fully.paid'] = df['not.fully.paid'].astype(bool)

    # Verify the changes
    print(df[['credit.policy', 'not.fully.paid']].dtypes)

    # List of features to analyze
    features_to_analyze = ['inq.last.6mths', 'delinq.2yrs', 'pub.rec']

    # Loop through each feature
    for feature in features_to_analyze:
        # Get unique values and their counts
        unique_values_counts = df[feature].value_counts()

        # Calculate the percentage per unique value
        percentages = (unique_values_counts / unique_values_counts.sum()) * 100

        # Sort the unique values in ascending order
        unique_values_counts = unique_values_counts.sort_index()
        percentages = percentages.sort_index()

        # Print the results
        print(f"Feature: {feature}")
        print("Unique Values\tCounts\tPercentage")
        for value, count, percentage in zip(unique_values_counts.index, unique_values_counts, percentages):
            print(f"{value}\t\t{count}\t\t{percentage:.2f}%")
        print("\n")

    # Define custom bin labels and bins
    custom_bins = [-1, 0, 2, 5, float('inf')]
    bin_labels = ['bin1', 'bin2', 'bin3', 'bin4']

    # Create a new column 'inq.last.6mths_bin' with custom bins
    df['inq.last.6mths_bin'] = pd.cut(df['inq.last.6mths'], bins=custom_bins, labels=bin_labels)

    # Calculate the distribution of the new discrete labels
    label_distribution = df['inq.last.6mths_bin'].value_counts()

    # Calculate the percentage distribution
    percentage_distribution = (label_distribution / label_distribution.sum()) * 100

    # Sort the values based on the custom bin labels
    label_distribution = label_distribution[bin_labels]
    percentage_distribution = percentage_distribution[bin_labels]

    # Print the distribution of new discrete labels with percentages
    print("Distribution of New Discrete Labels:")
    print("Label\tCounts\tPercentage")
    for label, count, percentage in zip(label_distribution.index, label_distribution, percentage_distribution):
        print(f"{label}\t{count}\t{percentage:.2f}%")

    # Define custom bin labels and bins
    custom_bins_delinq = [-1, 0, 1, float('inf')]
    bin_labels_delinq = ['bin1', 'bin2', 'bin3']

    # Create a new column 'delinq.2yrs_bin' with custom bins
    df['delinq.2yrs_bin'] = pd.cut(df['delinq.2yrs'], bins=custom_bins_delinq, labels=bin_labels_delinq)

    # Calculate the distribution of the new discrete labels
    label_distribution_delinq = df['delinq.2yrs_bin'].value_counts()

    # Calculate the percentage distribution
    percentage_distribution_delinq = (label_distribution_delinq / label_distribution_delinq.sum()) * 100

    # Sort the values based on the custom bin labels
    label_distribution_delinq = label_distribution_delinq[bin_labels_delinq]
    percentage_distribution_delinq = percentage_distribution_delinq[bin_labels_delinq]

    # Print the distribution of new discrete labels with percentages
    print("Distribution of New Discrete Labels for 'delinq.2yrs':")
    print("Label\tCounts\tPercentage")
    for label, count, percentage in zip(label_distribution_delinq.index, label_distribution_delinq, percentage_distribution_delinq):
        print(f"{label}\t{count}\t{percentage:.2f}%")

    # Define custom bin labels and bins
    custom_bins_pub_rec = [-1, 0, float('inf')]
    bin_labels_pub_rec = ['bin1', 'bin2']

    # Create a new column 'pub.rec_bin' with custom bins
    df['pub.rec_bin'] = pd.cut(df['pub.rec'], bins=custom_bins_pub_rec, labels=bin_labels_pub_rec)

    # Calculate the distribution of the new discrete labels
    label_distribution_pub_rec = df['pub.rec_bin'].value_counts()

    # Calculate the percentage distribution
    percentage_distribution_pub_rec = (label_distribution_pub_rec / label_distribution_pub_rec.sum()) * 100

    # Sort the values based on the custom bin labels
    label_distribution_pub_rec = label_distribution_pub_rec[bin_labels_pub_rec]
    percentage_distribution_pub_rec = percentage_distribution_pub_rec[bin_labels_pub_rec]

    # Print the distribution of new discrete labels with percentages
    print("Distribution of New Discrete Labels for 'pub.rec':")
    print("Label\tCounts\tPercentage")
    for label, count, percentage in zip(label_distribution_pub_rec.index, label_distribution_pub_rec, percentage_distribution_pub_rec):
        print(f"{label}\t{count}\t{percentage:.2f}%")

    # List of categorical features (including boolean features)
    categorical_features = ['purpose', 'credit.policy', 'not.fully.paid', 'inq.last.6mths_bin', 'delinq.2yrs_bin', 'pub.rec_bin']

    # Create subplots for each categorical feature
    plt.figure(figsize=(16, 10))

    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(2, 3, i)
        plt.title(f'Bar Chart for {feature}')

        if feature in ['credit.policy', 'not.fully.paid']:
            sns.countplot(data=df, x=feature, palette='Set2')
        else:
            sns.countplot(data=df, x=feature, palette='viridis')

        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_{FIG_COUNT}_categorical_bars.png"), dpi=300)
    plt.close()
    FIG_COUNT += 1

    # Filter and display records where 'revol.util' is greater than 100%
    invalid_revol_util_records = df[df['revol.util'] > 100]

    # Print the records
    print("Records with 'revol.util' > 100%:")
    print(invalid_revol_util_records)

    # Filter and change records where 'revol.util' is greater than 100% to 100%
    df.loc[df['revol.util'] > 100, 'revol.util'] = 100

    # Display the modified records
    print("Records after changing 'revol.util' values:")
    print(df[df['revol.util'] > 100])

    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64'])

    # Set the number of bins for the histograms
    num_bins = 30  # You can adjust this value based on your preference

    # Calculate the number of rows and columns for subplots
    num_features = len(numerical_columns.columns)
    num_rows = (num_features - 1) // 3 + 1
    num_cols = min(3, num_features)

    # Create subplots for each numerical feature
    plt.figure(figsize=(16, 4 * num_rows))  # Adjust the height based on the number of rows
    for i, column in enumerate(numerical_columns.columns, 1):
        plt.subplot(num_rows, num_cols, i)
        plt.title(f'Histogram for {column}')
        plt.hist(df[column], bins=num_bins, edgecolor='k')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_{FIG_COUNT}_numeric_histograms.png"), dpi=300)
    plt.close()
    FIG_COUNT += 1

    # Apply logarithm transformation to 'revol.bal'
    df['log_revol_bal'] = np.log1p(df['revol.bal'])

    # Create a histogram of the transformed values
    plt.figure(figsize=(8, 6))
    plt.title('Histogram of Log-transformed revol.bal')
    plt.hist(df['log_revol_bal'], bins=20, edgecolor='k')
    plt.xlabel('Log-transformed revol.bal')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(FIG_DIR, f"fig_{FIG_COUNT}_log_revol_bal.png"), dpi=300)
    plt.close()
    FIG_COUNT += 1

    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64'])

    # Calculate the correlation matrix using the Spearman method
    correlation_matrix = numerical_columns.corr(method='spearman')

    # Display the correlation matrix
    print("Spearman Correlation Matrix:")
    print(correlation_matrix)

    # Perform Spearman rank correlation test
    print("\nSpearman Rank Correlation Test:")
    for column1 in numerical_columns.columns:
        for column2 in numerical_columns.columns:
            if column1 != column2:
                correlation, p_value = stats.spearmanr(numerical_columns[column1], numerical_columns[column2])
                print(f"{column1} vs. {column2}:")
                print(f"  - Spearman correlation: {correlation:.4f}")
                print(f"  - p-value: {p_value:.4f}\n")

    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Spearman Correlation Heatmap')
    plt.savefig(os.path.join(FIG_DIR, f"fig_{FIG_COUNT}_spearman_heatmap.png"), dpi=300)
    plt.close()
    FIG_COUNT += 1

    # Perform one-hot encoding for the 'purpose' column
    one_hot_encoded_purpose = pd.get_dummies(df['purpose'], prefix='purpose')

    # Concatenate the one-hot encoded columns with the original DataFrame
    df = pd.concat([df, one_hot_encoded_purpose], axis=1)

    # Drop the original 'purpose' column, as it's no longer needed
    df.drop('purpose', axis=1, inplace=True)

    # Display the updated DataFrame with one-hot encoding
    print(df.head())

    df.info()

    # Drop the {revol.bal, inq.last.6mths, delinq.2yrs, pub.rec} columns from the DataFrame
    df.drop(['revol.bal', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec'], axis=1, inplace=True)

    # Display the updated DataFrame
    print(df.head())

    # List of categorical columns to label encode
    categorical_columns_to_encode = ['inq.last.6mths_bin', 'delinq.2yrs_bin', 'pub.rec_bin']

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to each specified column
    for column in categorical_columns_to_encode:
        df[column] = label_encoder.fit_transform(df[column])

    # Display the updated DataFrame with label encoding applied
    print(df.head())

    # Define your target variable (not.fully.paid) and features
    X = df.drop('not.fully.paid', axis=1)
    y = df['not.fully.paid']

    # Split the data into a training set (80%) and a test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the shapes of the resulting sets to verify the split
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Select only numerical columns in the training and test sets
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64'])
    numerical_columns_test = X_test.select_dtypes(include=['int64', 'float64'])

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train[numerical_columns.columns] = scaler.fit_transform(numerical_columns)

    # Transform the test data using the same scaler
    X_test[numerical_columns_test.columns] = scaler.transform(numerical_columns_test)

    X_train.head()
    X_test.head()

    # Apply SMOTE to oversample the minority class in the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", y_train_resampled.value_counts())
    os.makedirs("artifacts", exist_ok=True)

    # (keeping your original dumps exactly as you wrote them)
    joblib.dump(X_train_resampled, "artifacts/X_train.pkl")
    joblib.dump(X_test, "artifacts/X_test.pkl")
    joblib.dump(y_train_resampled, "artifacts/y_train.pkl")
    joblib.dump(y_test, "artifacts/y_test.pkl")

finally:
    # Always restore stdout properly
    try:
        sys.stdout.close()
    except Exception:
        pass
    sys.stdout = orig_stdout

print(f"Done. Figures saved in: {FIG_DIR}/")
print(f"Terminal output saved in: {terminal_path}")