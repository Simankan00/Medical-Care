import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load the training data
df = pd.read_csv("Training.csv")

# Prepare features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the model
svc_model = SVC(probability=True)
svc_model.fit(X, y_encoded)

# Create helper dictionaries
symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}
diseases_list = {idx: label for idx, label in enumerate(label_encoder.classes_)}

# Save the model and dictionaries
pickle.dump(svc_model, open("svc.pkl", "wb"))
pickle.dump(symptoms_dict, open("symptoms_dict.pkl", "wb"))
pickle.dump(diseases_list, open("diseases_list.pkl", "wb"))

print("✅ Model and dictionaries saved successfully.")
