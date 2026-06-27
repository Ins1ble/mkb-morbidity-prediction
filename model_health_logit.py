import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             brier_score_loss, mean_absolute_error)
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
data_file = 'dataset_for_model_health_self.csv'
df = pd.read_csv(data_file, encoding='utf-8-sig')
print(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов")

target_orig = 'Оценка состояния здоровья'
target = 'Высокая_самооценка'

# Бинаризация по медиане
threshold = df[target_orig].median()
df[target] = (df[target_orig] > threshold).astype(int)
print(f"Порог бинаризации (медиана): {threshold:.4f}")
print(f"\nРаспределение бинарной целевой переменной:")
print(df[target].value_counts())

# Признаки из исходной OLS-модели (18 признаков)
selected_factors = [
    'Материальное положение семьи', 'Материальное положение', 'ХЗ не имею', 'Семейное положение',
    'Оценка условий труда', 'Жилищно-бытовые условия', 'Частота отдыха', 'Образование',
    'Возраст начала приема алкоголя', 'Возраст постановки диагноза', 'Стаж работы на последнем месте работы',
    'Сопутствующие_болезни', 'Боль в пояснице', 'Возраст', 'Сахарный диабет',
    'Соблюдение диеты', 'Проводилась профилактика МКБ', 'Оценка виртуальных консультаций'
]

available_factors = [col for col in selected_factors if col in df.columns]
X = df[available_factors].copy()
y = df[target].copy()

mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]
print(f"\nИтоговый размер: {X.shape[0]} наблюдений, {X.shape[1]} признаков")

# ============================================================
# 2. СТАНДАРТИЗАЦИЯ
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ============================================================
# 3. ДИАГНОСТИКА МУЛЬТИКОЛЛИНЕАРНОСТИ
# ============================================================
# ============================================================
# 3.1 Корреляционная матрица (УЛУЧШЕННАЯ ЧИТАЕМОСТЬ ДЛЯ ПЕЧАТИ)
# ============================================================
# Сокращённые названия признаков
short_names = {
    'Материальное положение семьи': 'Мат. положение семьи',
    'Материальное положение': 'Материальное положение',
    'ХЗ не имею': 'ХЗ не имею',
    'Семейное положение': 'Семейное положение',
    'Оценка условий труда': 'Оценка условий труда',
    'Жилищно-бытовые условия': 'Жилищно-бытовые условия',
    'Частота отдыха': 'Частота отдыха',
    'Образование': 'Образование',
    'Возраст начала приема алкоголя': 'Возраст начала алкоголя',
    'Возраст постановки диагноза': 'Возраст постановки диагноза',
    'Стаж работы на последнем месте работы': 'Стаж на посл. месте работы',
    'Сопутствующие_болезни': 'Сопутствующие болезни',
    'Боль в пояснице': 'Боль в пояснице',
    'Возраст': 'Возраст',
    'Сахарный диабет': 'Сахарный диабет',
    'Соблюдение диеты': 'Соблюдение диеты',
    'Проводилась профилактика МКБ': 'Профилактика МКБ',
    'Оценка виртуальных консультаций': 'Оценка вирт. консультаций'
}

X_short = X_scaled.rename(columns=short_names)
corr_matrix = X_short.corr()

plt.figure(figsize=(22, 20))                         # холст прежний, 22×20
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 18},                  # крупные числа в ячейках
            xticklabels=True, yticklabels=True)
plt.xticks(fontsize=24, rotation=90, ha='center')    # ОЧЕНЬ крупные подписи
plt.yticks(fontsize=24, rotation=0)
plt.title('Корреляционная матрица предикторов (самооценка здоровья)', fontsize=28)
plt.tight_layout()
plt.savefig('health_self_correlation_matrix_large.png', dpi=600, bbox_inches='tight')
plt.show()

X_const = sm.add_constant(X_scaled)
vif_data = pd.DataFrame()
vif_data['Factor'] = X_scaled.columns
vif_data['VIF'] = [variance_inflation_factor(X_const.values, i+1) for i in range(len(X_scaled.columns))]
print("\nVIF для стандартизированных признаков:")
print(vif_data.sort_values('VIF', ascending=False))

# ============================================================
# 4. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# ============================================================
print("\n" + "="*60)
print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("="*60)

X_logit = sm.add_constant(X_scaled)
model_logit = sm.Logit(y, X_logit).fit(disp=False)
print(model_logit.summary())

y_pred_proba = model_logit.predict(X_logit)
y_pred_class = (y_pred_proba >= 0.5).astype(int)

acc_train = accuracy_score(y, y_pred_class)
roc_auc_train = roc_auc_score(y, y_pred_proba)
brier = brier_score_loss(y, y_pred_proba)
rmse_prob = np.sqrt(brier)
mae_prob = mean_absolute_error(y, y_pred_proba)
pseudo_r2 = model_logit.prsquared

print(f"\nМетрики на обучающей выборке:")
print(f"  Accuracy:          {acc_train:.4f}")
print(f"  ROC AUC:           {roc_auc_train:.4f}")
print(f"  Brier Score (MSE): {brier:.4f}")
print(f"  RMSE вероятностей: {rmse_prob:.4f}")
print(f"  MAE вероятностей:  {mae_prob:.4f}")
print(f"  McFadden R²:       {pseudo_r2:.4f}")
print("\nClassification Report (обучающая):")
print(classification_report(y, y_pred_class, target_names=['Низкая самооценка', 'Высокая самооценка']))

# ============================================================
# 5. КРОСС-ВАЛИДАЦИЯ (5-FOLD)
# ============================================================
print("\n" + "="*60)
print("КРОСС-ВАЛИДАЦИЯ (5-FOLD)")
print("="*60)

lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
cv_results = cross_validate(lr, X_scaled, y, cv=5,
                            scoring=['accuracy', 'roc_auc'])
acc_cv = cv_results['test_accuracy'].mean()
roc_auc_cv = cv_results['test_roc_auc'].mean()

print(f"Accuracy (CV): {acc_cv:.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"ROC AUC (CV):  {roc_auc_cv:.4f} ± {cv_results['test_roc_auc'].std():.4f}")

# ============================================================
# 6. ФОРМУЛА МОДЕЛИ
# ============================================================
print("\n" + "="*60)
print("ФОРМУЛА МОДЕЛИ")
print("="*60)
intercept = model_logit.params['const']
coefs = model_logit.params.drop('const')
formula = f"a = {intercept:.6f}"
for name, coef in coefs.items():
    formula += f" + {coef:.6f}*{name}"
print("Вероятность высокой самооценки P = 1 / (1 + exp(-a))")
print(f"где {formula}")

# ============================================================
# 7. СОХРАНЕНИЕ ОТЧЁТА
# ============================================================
with open('model_health_logit_final_report.txt', 'w', encoding='utf-8') as f:
    f.write("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ: ВЕРОЯТНОСТЬ ВЫСОКОЙ САМООЦЕНКИ ЗДОРОВЬЯ\n")
    f.write("="*60 + "\n\n")
    f.write(model_logit.summary().as_text())
    f.write(f"\n\nМетрики обучающей выборки:\n")
    f.write(f"Accuracy: {acc_train:.4f}\nROC AUC: {roc_auc_train:.4f}\n")
    f.write(f"Brier Score: {brier:.4f}\nRMSE prob: {rmse_prob:.4f}\nMAE prob: {mae_prob:.4f}\nMcFadden R²: {pseudo_r2:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y, y_pred_class, target_names=['Низкая самооценка', 'Высокая самооценка']))
    f.write(f"\nКросс-валидация (5 фолдов):\nAccuracy: {acc_cv:.4f} ± {cv_results['test_accuracy'].std():.4f}\nROC AUC: {roc_auc_cv:.4f} ± {cv_results['test_roc_auc'].std():.4f}\n")
    f.write("\nФормула модели:\n")
    f.write(f"P(Высокая самооценка) = 1 / (1 + exp(-a))\n{formula}\n")

print("\nГотово. Отчёт сохранён в 'model_health_logit_final_report.txt'")