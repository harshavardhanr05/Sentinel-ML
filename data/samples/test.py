import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load the data
df = pd.read_csv('heart_disease.csv')

# 2. Separate Features (X) and Target (y)
X = df.drop('Heart Disease Status', axis=1)
y = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})

# 3. Define Feature Groups based on the dataset structure
numeric_features = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 
                    'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar', 
                    'CRP Level', 'Homocysteine Level']

ordinal_features = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
# Setting the explicit order for the ordinal variables
ordinal_cats = [['Low', 'Medium', 'High']] * 4 

binary_features = ['Gender', 'Smoking', 'Family Heart Disease', 'Diabetes', 
                   'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol']

# 4. Build Preprocessing Steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=ordinal_cats))
])

binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='if_binary'))
])

# Combine all transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('bin', binary_transformer, binary_features)
    ])

# 5. Create the Full Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # class_weight='balanced' helps counteract any target imbalances
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 6. Train and Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print(classification_report(y_test, y_pred))