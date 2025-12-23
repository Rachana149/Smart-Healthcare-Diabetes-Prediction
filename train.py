import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("datasets.csv")


df = pd.get_dummies(df, drop_first=True)

X = df.drop("diabetes", axis=1)
y = df["diabetes"]


with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)


with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl, scaler.pkl, columns.pkl CREATED SUCCESSFULLY")
