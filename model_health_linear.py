import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
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

# Проверка наличия столбцов
missing_cols = [col for col in selected_factors if col not in df.columns]
if missing_cols:
    print(f"Предупреждение: отсутствуют столбцы {missing_cols}")
    selected_factors = [col for col in selected_factors if col in df.columns]

X = df[selected_factors].copy()
y = df[target].copy()

# Удаление строк с пропусками (на всякий случай)
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]
print(f"Итоговый размер: {X.shape[0]} наблюдений, {X.shape[1]} признаков")

# ============================================================
# 2. СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ============================================================
# 3. ДИАГНОСТИКА МУЛЬТИКОЛЛИНЕАРНОСТИ
# ============================================================
print("\n" + "=" * 60)
print("ДИАГНОСТИКА МУЛЬТИКОЛЛИНЕАРНОСТИ")
print("=" * 60)

# ============================================================
# 3.1 Корреляционная матрица (увеличенный шрифт для читаемости)
# ============================================================
corr_matrix = X_scaled.corr()
plt.figure(figsize=(18, 16))                        # увеличили холст
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 12},                # крупнее цифры в ячейках
            xticklabels=1, yticklabels=1)          # показать все подписи
plt.xticks(fontsize=12, rotation=45, ha='right')   # шрифт 12, наклон 45°
plt.yticks(fontsize=12, rotation=0)                # вертикальные подписи
plt.title('Корреляционная матрица предикторов (самооценка здоровья)', fontsize=14)
plt.tight_layout()
plt.savefig('health_self_correlation_matrix.png', dpi=300)
plt.show()

# 3.2 VIF
X_const = sm.add_constant(X_scaled)
vif_data = pd.DataFrame()
vif_data['Factor'] = X_scaled.columns
vif_data['VIF'] = [variance_inflation_factor(X_const.values, i + 1) for i in range(len(X_scaled.columns))]
print("\nVIF для стандартизированных признаков:")
print(vif_data.sort_values('VIF', ascending=False))

# 3.3 Сильно коррелирующие пары
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
if high_corr_pairs:
    print("\nПары признаков с |r| > 0.7:")
    for pair in high_corr_pairs:
        print(f"  {pair[0]} <-> {pair[1]}: r = {pair[2]:.3f}")

# ============================================================
# 4. ПОДХОД 1: OLS ПОСЛЕ УДАЛЕНИЯ ИЗБЫТОЧНОГО ПРИЗНАКА
# ============================================================
print("\n" + "=" * 60)
print("ПОДХОД 1: OLS ПОСЛЕ УДАЛЕНИЯ ИЗБЫТОЧНОГО ПРИЗНАКА")
print("=" * 60)

# Определяем, какой признак удалить (например, 'Материальное положение', если корреляция высока)
# Поскольку 'Материальное положение семьи' и 'Материальное положение' коррелируют ~0.68 (ниже 0.7),
# оставим оба, но если бы корреляция была >0.7, удалили бы один.
# В данном случае VIF в норме, поэтому просто повторим OLS без удаления (для демонстрации)
# Но чтобы сохранить структуру, условно удалим 'Материальное положение' и посмотрим эффект
# В реальной ситуации я бы оставил оба, т.к. VIF < 5.
X_reduced = X_scaled.drop(columns=['Материальное положение'])  # условно
X_red_const = sm.add_constant(X_reduced)
vif_red = pd.DataFrame()
vif_red['Factor'] = X_reduced.columns
vif_red['VIF'] = [variance_inflation_factor(X_red_const.values, i + 1) for i in range(len(X_reduced.columns))]
print("VIF после удаления 'Материальное положение':")
print(vif_red.sort_values('VIF', ascending=False))

model_red = sm.OLS(y, X_red_const).fit()
print("\nРезультаты OLS (стандартизированные коэффициенты):")
print(model_red.summary())

y_pred_red = model_red.predict(X_red_const)
r2_red = r2_score(y, y_pred_red)
rmse_red = np.sqrt(mean_squared_error(y, y_pred_red))
mae_red = mean_absolute_error(y, y_pred_red)
print(f"\nМетрики на всех данных: R² = {r2_red:.4f}, RMSE = {rmse_red:.4f}, MAE = {mae_red:.4f}")

# Кросс-валидация OLS reduced
lr_red = LinearRegression()
cv_results_red = cross_validate(lr_red, X_reduced, y, cv=5,
                                scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'])
r2_cv_red = cv_results_red['test_r2'].mean()
rmse_cv_red = -cv_results_red['test_neg_root_mean_squared_error'].mean()
mae_cv_red = -cv_results_red['test_neg_mean_absolute_error'].mean()
print(f"\nКросс-валидация (5 фолдов): R² = {r2_cv_red:.4f}, RMSE = {rmse_cv_red:.4f}, MAE = {mae_cv_red:.4f}")

# ============================================================
# 5. ПОДХОД 2: RIDGE-РЕГРЕССИЯ (ВСЕ ПРИЗНАКИ)
# ============================================================
print("\n" + "=" * 60)
print("ПОДХОД 2: RIDGE-РЕГРЕССИЯ (L2-регуляризация)")
print("=" * 60)

alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_scaled, y)
print(f"Оптимальное значение alpha: {ridge_cv.alpha_:.4f}")

ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_scaled, y)

coef_ridge = pd.Series(ridge.coef_, index=X_scaled.columns)
print("\nКоэффициенты Ridge (стандартизированные):")
print(coef_ridge.sort_values(ascending=False))

y_pred_ridge = ridge.predict(X_scaled)
r2_ridge = r2_score(y, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y, y_pred_ridge))
mae_ridge = mean_absolute_error(y, y_pred_ridge)
print(f"\nМетрики на всех данных: R² = {r2_ridge:.4f}, RMSE = {rmse_ridge:.4f}, MAE = {mae_ridge:.4f}")

ridge_fixed = Ridge(alpha=ridge_cv.alpha_)
cv_results_ridge = cross_validate(ridge_fixed, X_scaled, y, cv=5,
                                  scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'])
r2_cv_ridge = cv_results_ridge['test_r2'].mean()
rmse_cv_ridge = -cv_results_ridge['test_neg_root_mean_squared_error'].mean()
mae_cv_ridge = -cv_results_ridge['test_neg_mean_absolute_error'].mean()
print(f"\nКросс-валидация (5 фолдов): R² = {r2_cv_ridge:.4f}, RMSE = {rmse_cv_ridge:.4f}, MAE = {mae_cv_ridge:.4f}")

# ============================================================
# 6. СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================
print("\n" + "=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)
comparison_df = pd.DataFrame({
    'Модель': ['OLS (удалён признак)', 'Ridge (все признаки)'],
    'R² (CV)': [r2_cv_red, r2_cv_ridge],
    'RMSE (CV)': [rmse_cv_red, rmse_cv_ridge],
    'MAE (CV)': [mae_cv_red, mae_cv_ridge]
})
print(comparison_df.to_string(index=False))

# Важность признаков Ridge
plt.figure(figsize=(10, 6))
coef_sorted = coef_ridge.abs().sort_values(ascending=True)
plt.barh(coef_sorted.index, coef_sorted.values, color='skyblue')
plt.xlabel('Абсолютное значение стандартизированного коэффициента')
plt.title('Важность признаков в Ridge-регрессии (самооценка здоровья)')
plt.tight_layout()
plt.savefig('health_self_ridge_importance.png', dpi=300)
plt.show()

# ============================================================
# 7. ДИАГНОСТИКА ОСТАТКОВ ДЛЯ RIDGE
# ============================================================
residuals = y - y_pred_ridge
fitted = y_pred_ridge

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Диагностика остатков Ridge-модели (самооценка здоровья)', fontsize=14, weight='bold')

axes[0, 0].scatter(fitted, y, alpha=0.5)
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[0, 0].set_xlabel('Предсказанные')
axes[0, 0].set_ylabel('Фактические')
axes[0, 0].set_title('Предсказанные vs фактические')

axes[0, 1].scatter(fitted, residuals, alpha=0.5)
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Предсказанные')
axes[0, 1].set_ylabel('Остатки')
axes[0, 1].set_title('Остатки vs предсказанные')

stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q plot остатков')

sns.histplot(residuals, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Распределение остатков')
axes[1, 1].set_xlabel('Остатки')

plt.tight_layout()
plt.savefig('health_self_ridge_residuals.png', dpi=300)
plt.show()

# ============================================================
# 8. СОХРАНЕНИЕ ОТЧЁТА
# ============================================================
with open('model_health_self_final_report.txt', 'w', encoding='utf-8') as f:
    f.write("ФИНАЛЬНЫЙ ОТЧЁТ ПО МОДЕЛИ САМООЦЕНКИ ЗДОРОВЬЯ\n")
    f.write("=" * 60 + "\n\n")
    f.write("Диагностика мультиколлинеарности:\n")
    f.write(f"Максимальный VIF: {vif_data['VIF'].max():.2f}\n")
    if high_corr_pairs:
        f.write("Сильно коррелирующие пары:\n")
        for pair in high_corr_pairs:
            f.write(f"  {pair[0]} <-> {pair[1]}: r = {pair[2]:.3f}\n")
    f.write("\n")

    f.write("Подход 1: OLS после удаления 'Материальное положение'\n")
    f.write(f"  R² (все данные): {r2_red:.4f}\n")
    f.write(f"  R² (CV 5-fold): {r2_cv_red:.4f}\n")
    f.write(f"  RMSE (CV): {rmse_cv_red:.4f}\n")
    f.write(f"  MAE (CV): {mae_cv_red:.4f}\n\n")

    f.write("Подход 2: Ridge-регрессия (все признаки)\n")
    f.write(f"  Оптимальное alpha: {ridge_cv.alpha_:.4f}\n")
    f.write(f"  R² (все данные): {r2_ridge:.4f}\n")
    f.write(f"  R² (CV 5-fold): {r2_cv_ridge:.4f}\n")
    f.write(f"  RMSE (CV): {rmse_cv_ridge:.4f}\n")
    f.write(f"  MAE (CV): {mae_cv_ridge:.4f}\n\n")

    f.write("Стандартизированные коэффициенты Ridge:\n")
    for name, coef in coef_ridge.sort_values(ascending=False).items():
        f.write(f"  {name:45}: {coef:+.4f}\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("РЕКОМЕНДАЦИЯ: ")
    if r2_cv_ridge >= r2_cv_red:
        f.write("Ridge-регрессия показывает лучшую обобщающую способность. Рекомендуется Ridge.\n")
    else:
        f.write("Модель с удалением признака показывает лучший результат. Рекомендуется OLS reduced.\n")

print("\nГотово. Отчёт сохранён в 'model_health_self_final_report.txt'")