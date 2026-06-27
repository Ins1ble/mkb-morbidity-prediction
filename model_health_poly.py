import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV, RidgeCV, Ridge
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
data_file = 'dataset_for_model_health_self.csv'
df = pd.read_csv(data_file, encoding='utf-8-sig')
print(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов")

target = 'Оценка состояния здоровья'

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
print(f"Итоговый размер: {X.shape[0]} наблюдений, {X.shape[1]} признаков")

# ============================================================
# 2. СТАНДАРТИЗАЦИЯ
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ============================================================
# 3. ДИАГНОСТИКА МУЛЬТИКОЛЛИНЕАРНОСТИ
# ============================================================
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
# 4. ГЕНЕРАЦИЯ ПОЛИНОМИАЛЬНЫХ ПРИЗНАКОВ (СТЕПЕНЬ 2)
# ============================================================
print("\n" + "="*60)
print("РАСШИРЕНИЕ ПРИЗНАКОВ ПОЛИНОМАМИ СТЕПЕНИ 2")
print("="*60)

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_scaled)
poly_feature_names = poly.get_feature_names_out(X_scaled.columns)
X_poly = pd.DataFrame(X_poly, columns=poly_feature_names, index=X_scaled.index)
print(f"Исходных признаков: {X_scaled.shape[1]}, после расширения: {X_poly.shape[1]}")

# ============================================================
# 5. ОТБОР ПРИЗНАКОВ ЧЕРЕЗ ELASTICNET (L1+L2)
# ============================================================
print("\n" + "="*60)
print("ОТБОР ПРИЗНАКОВ (ElasticNet)")
print("="*60)

enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1], cv=5, random_state=42, max_iter=5000)
enet_cv.fit(X_poly, y)
best_alpha = enet_cv.alpha_
best_l1 = enet_cv.l1_ratio_
print(f"Оптимальные параметры ElasticNet: alpha={best_alpha:.4f}, l1_ratio={best_l1:.2f}")

selector = SelectFromModel(enet_cv, prefit=True)  # используем уже обученный enet_cv
selected_features = X_poly.columns[selector.get_support()]
print(f"Отобрано признаков: {len(selected_features)} из {X_poly.shape[1]}")

X_selected = X_poly[selected_features]

# ============================================================
# 6. ФИНАЛЬНАЯ RIDGE-РЕГРЕССИЯ НА ОТОБРАННЫХ ПРИЗНАКАХ
# ============================================================
print("\n" + "="*60)
print("ФИНАЛЬНАЯ RIDGE-РЕГРЕССИЯ")
print("="*60)

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_selected, y)
best_alpha_ridge = ridge_cv.alpha_
print(f"Оптимальное alpha для Ridge: {best_alpha_ridge:.4f}")

ridge = Ridge(alpha=best_alpha_ridge)
ridge.fit(X_selected, y)

coef_ridge = pd.Series(ridge.coef_, index=X_selected.columns).sort_values(ascending=False)

y_pred = ridge.predict(X_selected)
mse_train = mean_squared_error(y, y_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y, y_pred)
mae_train = mean_absolute_error(y, y_pred)
adj_r2_train = 1 - (1 - r2_train) * (len(y) - 1) / (len(y) - X_selected.shape[1] - 1)

print(f"\nМетрики на обучающей выборке:")
print(f"  R²:            {r2_train:.4f}")
print(f"  Adjusted R²:   {adj_r2_train:.4f}")
print(f"  RMSE:          {rmse_train:.4f}")
print(f"  MAE:           {mae_train:.4f}")

# ============================================================
# 7. КРОСС-ВАЛИДАЦИЯ
# ============================================================
print("\n" + "="*60)
print("КРОСС-ВАЛИДАЦИЯ (5-FOLD)")
print("="*60)

cv_results = cross_validate(ridge, X_selected, y, cv=5,
                            scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'])
r2_cv = cv_results['test_r2'].mean()
rmse_cv = -cv_results['test_neg_root_mean_squared_error'].mean()
mae_cv = -cv_results['test_neg_mean_absolute_error'].mean()

print(f"R² (CV):   {r2_cv:.4f} ± {cv_results['test_r2'].std():.4f}")
print(f"RMSE (CV): {rmse_cv:.4f} ± {cv_results['test_neg_root_mean_squared_error'].std():.4f}")
print(f"MAE (CV):  {mae_cv:.4f} ± {cv_results['test_neg_mean_absolute_error'].std():.4f}")

# ============================================================
# 8. ГРАФИК ВАЖНОСТИ ПРИЗНАКОВ
# ============================================================
# ============================================================
# 8. ГРАФИК ВАЖНОСТИ ПРИЗНАКОВ (УЛУЧШЕННАЯ ЧИТАЕМОСТЬ, "×" ТОЛЬКО МЕЖДУ ПРИЗНАКАМИ)
# ============================================================
import re

# Исходные названия признаков (используем X.columns из загруженных данных)
original_features = list(X.columns)


def make_readable(name):
    """
    Преобразует имя монома в читаемое:
    - линейный член: оставляет как есть,
    - квадрат: 'Признак²',
    - взаимодействие: 'Признак1 × Признак2'.
    """
    # Проверяем, является ли это простым признаком или его квадратом
    if name in original_features:
        return name
    if name.endswith('^2'):
        base = name[:-2]
        if base in original_features:
            return base + '²'

    # Для взаимодействий ищем комбинацию двух исходных признаков
    # Перебираем все пары, проверяем, можно ли разбить строку на две части,
    # каждая из которых совпадает с каким-либо исходным признаком.
    for i in range(len(original_features)):
        f1 = original_features[i]
        if name.startswith(f1):
            remainder = name[len(f1):].strip()
            if remainder in original_features:
                return f1 + ' × ' + remainder
            # также проверяем, может быть второй признак содержит пробелы и его "остаток" совпадает после удаления пробела
            # но поскольку мы перебираем все признаки, то если remainder совпадает с каким-то признаком, ок.
            for j in range(len(original_features)):
                if remainder == original_features[j]:
                    return f1 + ' × ' + remainder
    # Если не удалось разобрать, возвращаем как есть (на всякий случай)
    return name


# Применяем преобразование ко всем названиям мономов
readable_index = [make_readable(name) for name in coef_ridge.index]

top_coef = pd.Series(coef_ridge.values, index=readable_index)
top_coef = pd.concat([top_coef.head(15), top_coef.tail(15)])

plt.figure(figsize=(16, 14))
colors = ['green' if c > 0 else 'red' for c in top_coef.values]
plt.barh(top_coef.index, top_coef.values, color=colors)
plt.xlabel('Стандартизированный коэффициент', fontsize=16)
plt.title('Топ-30 признаков по важности (Ridge)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('health_ridge_importance_optimized.png', dpi=600)
plt.show()
# ============================================================
# 9. СОХРАНЕНИЕ ОТЧЁТА (с пояснениями для каждого монома)
# ============================================================
def describe_monom(name: str, original_columns: list) -> str:
    """
    Возвращает описание монома по его имени, полученному от PolynomialFeatures.
    Примеры:
      'Возраст'          -> 'исходный признак «Возраст»'
      'Возраст^2'        -> 'квадрат признака «Возраст» (нелинейный эффект)'
      'Возраст Стаж'     -> 'взаимодействие (произведение) признаков «Возраст» и «Стаж работы на последнем месте работы»'
    """
    parts = name.split()
    if len(parts) == 1:
        part = parts[0]
        if '^' in part:
            base, exp = part.split('^')
            return f"квадрат признака «{base}» (нелинейный эффект)"
        else:
            return f"исходный признак «{part}»"
    else:
        interactions = []
        for p in parts:
            if '^' in p:
                base, _ = p.split('^')
                interactions.append(f"«{base}» в квадрате")
            else:
                interactions.append(f"«{p}»")
        if len(interactions) == 2:
            return f"взаимодействие (произведение) признаков {interactions[0]} и {interactions[1]}"
        else:
            return f"взаимодействие признаков: {' * '.join(interactions)}"

# Генерируем описания для всех коэффициентов (порядок сохранён)
descriptions = [describe_monom(name, X.columns.tolist()) for name in coef_ridge.index]

with open('model_health_ridge_optimized_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОПТИМИЗИРОВАННАЯ RIDGE-РЕГРЕССИЯ (САМООЦЕНКА ЗДОРОВЬЯ)\n")
    f.write("="*60 + "\n\n")
    f.write(f"Отобрано признаков (ElasticNet): {len(selected_features)}\n")
    f.write(f"Оптимальное alpha Ridge: {best_alpha_ridge:.4f}\n\n")
    f.write("Метрики обучающей выборки:\n")
    f.write(f"R²: {r2_train:.4f}\nAdjusted R²: {adj_r2_train:.4f}\nRMSE: {rmse_train:.4f}\nMAE: {mae_train:.4f}\n\n")
    f.write("Кросс-валидация (5 фолдов):\n")
    f.write(f"R²:   {r2_cv:.4f} ± {cv_results['test_r2'].std():.4f}\n")
    f.write(f"RMSE: {rmse_cv:.4f} ± {cv_results['test_neg_root_mean_squared_error'].std():.4f}\n")
    f.write(f"MAE:  {mae_cv:.4f} ± {cv_results['test_neg_mean_absolute_error'].std():.4f}\n\n")

    f.write("КОЭФФИЦИЕНТЫ МОДЕЛИ (отсортированы по убыванию)\n")
    f.write("-"*80 + "\n")
    f.write(f"{'№':3} {'Моном':45} {'Коэффициент':>12}   Описание\n")
    f.write("-"*80 + "\n")

    for idx, (name, coef, desc) in enumerate(zip(coef_ridge.index, coef_ridge.values, descriptions), 1):
        f.write(f"{idx:3} {name:45} {coef:+12.6f}   {desc}\n")

    f.write("\n" + "="*60 + "\n")
    f.write("Примечание: все признаки стандартизированы (среднее=0, ст.откл.=1)\n")

print("\nГотово. Отчёт с описаниями мономов сохранён в 'model_health_ridge_optimized_report.txt'")