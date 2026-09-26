import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('E-commerce_Customer_Segmentation_2026.csv')
df['high_churn_risk'] = df['churn_risk_category'].isin(['High', 'Very High']).astype(int)

FEATURE_ORDER = ['employment_type', 'purchase_frequency', 'days_since_last_purchase',
                  'shopping_channel', 'return_count', 'complaint_count', 'satisfaction_score']
CAT_COLS = ['employment_type', 'purchase_frequency', 'shopping_channel']

X = df[FEATURE_ORDER].copy()
y = df['high_churn_risk']

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    print(col, dict(zip(le.classes_, le.transform(le.classes_))))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

candidates = {
    'RandomForest_balanced': RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1),
    'HistGB': HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=300, random_state=42),
}

results = {}
trained = {}
for name, clf in candidates.items():
    if name == 'HistGB':
        # class weighting via sample_weight
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight('balanced', y_train)
        clf.fit(x_train, y_train, sample_weight=sw)
    else:
        clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)[:, 1]
    results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, proba),
    }
    trained[name] = clf
    print(name, results[name])

best_name = max(results, key=lambda n: results[n]['roc_auc'])
print('BEST:', best_name)
best_model = trained[best_name]
print(classification_report(y_test, best_model.predict(x_test), target_names=['Not High Risk','High Risk']))

bundle = {
    'model': best_model,
    'model_name': best_name,
    'encoders': encoders,
    'feature_order': FEATURE_ORDER,
    'metrics': results[best_name],
}
with open('churn_pipeline.pkl', 'wb') as f:
    pickle.dump(bundle, f)
print('Saved churn_pipeline.pkl')
