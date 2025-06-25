import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Train model
def train_model(df):
    # Convert Pass/Fail to binary (Pass = 1, Fail = 0)
    df['Label'] = LabelEncoder().fit_transform(df['Label'])

    X = df[['SM', 'EM']]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred,zero_division=0))

    return clf

def predict_new(clf, sm, em):
    input_df = pd.DataFrame({'SM': [sm], 'EM': [em]})  # Use same feature names as training
    prob = clf.predict_proba(input_df)[0][1]  # Probability of passing
    label = clf.predict(input_df)[0]
    result = "Pass" if label == 1 else "Fail"
    print(f"\nPrediction: {result} (Confidence: {prob:.2f})")

def predict_from_file(clf, input_file, output_file="predictions.csv"):
    # Load new data (no labels)
    new_data = pd.read_csv(input_file)

    # Predict
    predictions = clf.predict(new_data)
    probabilities = clf.predict_proba(new_data)[:, 1]  # Probability of Pass

    # Convert numeric prediction back to label
    label_map = {1: "Pass", 0: "Fail"}
    results = pd.DataFrame(new_data)
    results["Prediction"] = [label_map[p] for p in predictions]
    results["Confidence"] = probabilities.round(2)

    # Save to file
    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print(results)


# Main program
if __name__ == "__main__":
     # Step 1: Train model from historical data
    df = load_data("Marks.csv")  # Historical training data
    model = train_model(df)

    # Step 2: Predict on new students
    predict_from_file(model, "GuessMarks.csv")