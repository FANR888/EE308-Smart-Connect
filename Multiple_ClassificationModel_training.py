# ==================== GLOBAL OUTPUT DIRECTORY ====================
OUTPUT_DIR = "./training_outputs"   # <-- modify this path as needed

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ==================== MAIN TRAINING CODE ====================

# Load data
df = pd.read_csv('./features_robust.csv')
X = df.drop(columns=['label', 'file']).values
y = LabelEncoder().fit_transform(df['label'].values)
le = LabelEncoder().fit(df['label'])
class_names = le.classes_

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
print(f"Total samples: {len(df)}")
print(f"Number of classes: {len(class_names)}")
print(f"Number of features: {X.shape[1]}")
print(f"Samples per class: {len(df) / len(class_names):.1f} (average)")
print(f"Samples per feature ratio: {len(df) / X.shape[1]:.2f}")
print("=" * 70 + "\n")

# Split data: 56% train, 24% val, 20% test
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.3, stratify=y_train_temp, random_state=42)

print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")

# Feature scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Test with and without PCA
USE_PCA = False

if USE_PCA:
    print("\n" + "=" * 70)
    print("APPLYING PCA FOR DIMENSIONALITY REDUCTION")
    print("=" * 70)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_s = pca.fit_transform(X_train_s)
    X_val_s = pca.transform(X_val_s)
    X_test_s = pca.transform(X_test_s)
    print(f"Original features: {X.shape[1]}")
    print(f"PCA reduced features to: {X_train_s.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print("=" * 70 + "\n")
else:
    pca = None
    print(f"\nUsing all {X.shape[1]} features (no PCA)\n")

print("=" * 70)
print("MODEL TRAINING & EVALUATION")
print("=" * 70)

# Dictionary to store results
results = {}

# Model 1: Constrained Random Forest
print("\n1. Training Constrained Random Forest...")
rf_constrained = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_constrained.fit(X_train_s, y_train)

y_train_pred_rf = rf_constrained.predict(X_train_s)
y_val_pred_rf = rf_constrained.predict(X_val_s)
y_test_pred_rf = rf_constrained.predict(X_test_s)

results['Constrained RF'] = {
    'train': accuracy_score(y_train, y_train_pred_rf),
    'val': accuracy_score(y_val, y_val_pred_rf),
    'test': accuracy_score(y_test, y_test_pred_rf)
}
print(f"   Train Acc: {results['Constrained RF']['train']:.4f}")
print(f"   Val Acc:   {results['Constrained RF']['val']:.4f}")
print(f"   Test Acc:  {results['Constrained RF']['test']:.4f}")

# Model 2: Gradient Boosting
print("\n2. Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_s, y_train)

y_train_pred_gb = gb.predict(X_train_s)
y_val_pred_gb = gb.predict(X_val_s)
y_test_pred_gb = gb.predict(X_test_s)

results['Gradient Boosting'] = {
    'train': accuracy_score(y_train, y_train_pred_gb),
    'val': accuracy_score(y_val, y_val_pred_gb),
    'test': accuracy_score(y_test, y_test_pred_gb)
}
print(f"   Train Acc: {results['Gradient Boosting']['train']:.4f}")
print(f"   Val Acc:   {results['Gradient Boosting']['val']:.4f}")
print(f"   Test Acc:  {results['Gradient Boosting']['test']:.4f}")

# Model 3: SVM with RBF kernel
print("\n3. Training SVM...")
svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    probability=True,  # Enable probability estimates for top-5 predictions
    random_state=42
)
svm.fit(X_train_s, y_train)

y_train_pred_svm = svm.predict(X_train_s)
y_val_pred_svm = svm.predict(X_val_s)
y_test_pred_svm = svm.predict(X_test_s)

results['SVM'] = {
    'train': accuracy_score(y_train, y_train_pred_svm),
    'val': accuracy_score(y_val, y_val_pred_svm),
    'test': accuracy_score(y_test, y_test_pred_svm)
}
print(f"   Train Acc: {results['SVM']['train']:.4f}")
print(f"   Val Acc:   {results['SVM']['val']:.4f}")
print(f"   Test Acc:  {results['SVM']['test']:.4f}")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY: ALL MODELS")
print("=" * 70)
results_df = pd.DataFrame(results).T
results_df['Train-Val Gap'] = results_df['train'] - results_df['val']
print(results_df.to_string())
print("=" * 70)

# Select best model based on validation accuracy
best_model_name = results_df['val'].idxmax()
print(f"\nBest model: {best_model_name}")

# Use the best model for final evaluation
if best_model_name == 'Constrained RF':
    best_model = rf_constrained
    y_test_pred_best = y_test_pred_rf
elif best_model_name == 'Gradient Boosting':
    best_model = gb
    y_test_pred_best = y_test_pred_gb
else:
    best_model = svm
    y_test_pred_best = y_test_pred_svm

# Cross-validation on best model
print("\n" + "=" * 70)
print(f"CROSS-VALIDATION (5-Fold) on {best_model_name}")
print("=" * 70)
cv_scores = cross_val_score(best_model, X_train_s, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save model and preprocessing objects
print("\nSaving model and preprocessing objects...")
joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
if pca is not None:
    joblib.dump(pca, os.path.join(OUTPUT_DIR, 'pca.pkl'))
print("Saved: best_model.pkl, scaler.pkl, label_encoder.pkl" +
      (", pca.pkl" if pca is not None else ""))

# Generate reports and visualizations
print(f"\nGenerating detailed reports for {best_model_name}...")
test_report = classification_report(y_test, y_test_pred_best,
                                    target_names=class_names,
                                    output_dict=True)
test_report_df = pd.DataFrame(test_report).transpose()
test_report_df.to_csv(os.path.join(OUTPUT_DIR, 'test_classification_report.csv'))

cm = confusion_matrix(y_test, y_test_pred_best)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix_test.csv'))

accuracy_summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Train_Accuracy': [results[m]['train'] for m in results],
    'Val_Accuracy': [results[m]['val'] for m in results],
    'Test_Accuracy': [results[m]['test'] for m in results]
})
accuracy_summary.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

if best_model_name == 'Constrained RF':
    feature_importance = pd.DataFrame({
        'Feature_Index': range(X_train_s.shape[1]),
        'Importance': rf_constrained.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

# Visualizations
plt.figure(figsize=(16, 14))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name} (Test Set)', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_best_model.png'), dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix_best_model.png'")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
x_pos = np.arange(len(results))
width = 0.25
ax1.bar(x_pos - width, [results[m]['train'] for m in results],
        width, label='Train', alpha=0.8)
ax1.bar(x_pos, [results[m]['val'] for m in results],
        width, label='Validation', alpha=0.8)
ax1.bar(x_pos + width, [results[m]['test'] for m in results],
        width, label='Test', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results.keys(), rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
gaps = [results[m]['train'] - results[m]['val'] for m in results]
colors = ['red' if g > 0.15 else 'orange' if g > 0.05 else 'green' for g in gaps]
ax2.bar(results.keys(), gaps, color=colors, alpha=0.7)
ax2.set_xlabel('Model')
ax2.set_ylabel('Train-Val Accuracy Gap')
ax2.set_title('Overfitting Analysis')
ax2.axhline(y=0.05, color='green', linestyle='--', linewidth=1, label='Good (<5%)')
ax2.axhline(y=0.15, color='orange', linestyle='--', linewidth=1, label='Warning (>15%)')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_charts.png'), dpi=300, bbox_inches='tight')
print("Model comparison charts saved as 'model_comparison_charts.png'")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\nFiles generated:")
print("  - best_model.pkl, scaler.pkl, label_encoder.pkl")
print("  - test_classification_report.csv")
print("  - confusion_matrix_test.csv")
print("  - model_comparison.csv")
print("  - feature_importance.csv (if RF was best)")
print("  - confusion_matrix_best_model.png")
print("  - model_comparison_charts.png")
print("=" * 70)
