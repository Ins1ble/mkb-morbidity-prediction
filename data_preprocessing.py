import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================
print("=" * 60)
print("ЭТАП 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

file_path = 'Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Кодированная', header=0)
print(f"Исходный размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")

# Оставляем только пациентов основной группы (МКБ)
df_main = df[df['Группа'] == 1].copy()
print(f"Пациентов с МКБ (основная группа): {df_main.shape[0]}")

# ============================================================
# 2. СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ (НАСЛЕДСТВЕННОСТЬ, ДЛИТЕЛЬНОСТЬ МКБ, СОПУТСТВУЮЩИЕ)
# ============================================================
print("\nСоздаем новые признаки...")

# Наследственность по линиям
inheritance_cols = {
    'МКБ дедушка/бабушка': 'Наследственность_дедушка',
    'МКБ тетя/дядя': 'Наследственность_тетя',
    'МКБ двоюродный брат/сестра': 'Наследственность_двоюродные',
    'МКБ папа/мама': 'Наследственность_родители',
    'МКБ брат/сестра': 'Наследственность_сиблинги',
    'МКБ мой ребенок': 'Наследственность_дети'
}
for col, new_name in inheritance_cols.items():
    if col in df_main.columns:
        df_main[new_name] = pd.to_numeric(df_main[col], errors='coerce').fillna(0)
    else:
        df_main[new_name] = 0
print(f"Создано признаков наследственности: {len(inheritance_cols)}")

# Суммарный индекс наследственности
inheritance_sum_cols = list(inheritance_cols.values())
df_main['Наследственность_сумма'] = df_main[inheritance_sum_cols].sum(axis=1)

# Длительность МКБ
if 'Возраст постановки диагноза' in df_main.columns and 'Возраст' in df_main.columns:
    age_diagnosis = pd.to_numeric(df_main['Возраст постановки диагноза'], errors='coerce')
    current_age = pd.to_numeric(df_main['Возраст'], errors='coerce')
    df_main['Длительность_МКБ'] = current_age - age_diagnosis
    df_main.loc[df_main['Длительность_МКБ'] < 0, 'Длительность_МКБ'] = np.nan
else:
    df_main['Длительность_МКБ'] = np.nan

# Количество сопутствующих болезней
comorbidity_cols = ['ХЗ сердечно-сосудистой сист', 'ХЗ эндокринной сист', 'Сахарный диабет']
available_comorbidity = [col for col in comorbidity_cols if col in df_main.columns]
if available_comorbidity:
    df_main['Сопутствующие_болезни'] = df_main[available_comorbidity].sum(axis=1)
else:
    df_main['Сопутствующие_болезни'] = 0

# ============================================================
# 3. ОТБОР ФАКТОРОВ ДЛЯ МОДЕЛИ САМООЦЕНКИ ЗДОРОВЬЯ
# ============================================================
print("\nОтбираем факторы для модели самооценки здоровья...")

factors_health_model = [
    'Материальное положение семьи', 'Материальное положение', 'ХЗ не имею', 'Семейное положение',
    'Оценка условий труда', 'Жилищно-бытовые условия', 'Частота отдыха', 'Образование',
    'Возраст начала приема алкоголя', 'Возраст постановки диагноза', 'Стаж работы на последнем месте работы',
    'Сопутствующие_болезни', 'Боль в пояснице', 'Возраст', 'Сахарный диабет',
    'Соблюдение диеты', 'Проводилась профилактика МКБ', 'Оценка виртуальных консультаций'
]

# Оставляем только те, что есть в данных
available_factors = [col for col in factors_health_model if col in df_main.columns]
missing_factors = [col for col in factors_health_model if col not in df_main.columns]
print(f"Доступно факторов: {len(available_factors)} из {len(factors_health_model)}")
if missing_factors:
    print(f"ВНИМАНИЕ: Отсутствуют в данных: {missing_factors}")

# Целевая переменная
target_health = 'Оценка состояния здоровья'
if target_health not in df_main.columns:
    print(f"ОШИБКА: Не найден столбец '{target_health}'")
    exit()

# ============================================================
# 4. ОБРАБОТКА ДАННЫХ С MICE ДЛЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ============================================================
print("\n" + "=" * 60)
print("ОБРАБОТКА ПРОПУСКОВ МЕТОДОМ MICE")
print("=" * 60)

target_name = target_health
all_cols = available_factors + [target_name]
df_target = df_main[all_cols].copy()

# Удаляем дубликаты имён столбцов
if df_target.columns.duplicated().any():
    df_target = df_target.loc[:, ~df_target.columns.duplicated()]

# Приводим все столбцы к числовому типу
for col in df_target.columns:
    df_target[col] = pd.to_numeric(df_target[col], errors='coerce')

# Удаляем строки с пропусками в целевой переменной
initial_rows = len(df_target)
df_target = df_target.dropna(subset=[target_name])
print(f"Удалено строк с пропусками в {target_name}: {initial_rows - len(df_target)}")
print(f"Осталось пациентов: {len(df_target)}")

# Определяем список предикторов
predictors = [col for col in df_target.columns if col != target_name]

# Анализируем пропуски в предикторах
missing_pct = df_target[predictors].isnull().sum() / len(df_target) * 100
missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(missing_cols) > 0:
    print("\nСтолбцы с пропусками (%):")
    for col, pct in missing_cols.items():
        print(f"  {col}: {pct:.1f}%")

# Удаляем столбцы с >50% пропусков
cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
if cols_to_drop:
    print(f"\nУдаляем столбцы с >50% пропусков: {cols_to_drop}")
    df_target = df_target.drop(columns=cols_to_drop)
    predictors = [col for col in predictors if col in df_target.columns]

# MICE – множественная импутация
print("\nВыполнение множественной импутации (MICE)...")
df_impute = df_target[predictors].copy()
imputer = IterativeImputer(max_iter=10, random_state=42)
imputed_array = imputer.fit_transform(df_impute)
df_target[predictors] = imputed_array

# Проверяем, что пропусков не осталось
remaining_nulls = df_target.isnull().sum().sum()
if remaining_nulls > 0:
    print(f"\nВНИМАНИЕ: После MICE осталось пропусков: {remaining_nulls}")
    df_target = df_target.dropna()
else:
    print(f"\nПропусков не осталось, сохранено {len(df_target)} пациентов")

# ============================================================
# 5. СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ
# ============================================================
print("\nСтандартизация признаков...")
scaler = StandardScaler()
df_target[predictors] = scaler.fit_transform(df_target[predictors])

# ============================================================
# 6. СОХРАНЕНИЕ ОЧИЩЕННОГО НАБОРА ДАННЫХ
# ============================================================
output_filename = 'dataset_for_model_health_self.csv'
df_target.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Очищенный датасет сохранен: {output_filename}")
print(f"Размер: {df_target.shape[0]} строк, {df_target.shape[1]} столбцов")

# ============================================================
# 7. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ И ЭКСПОРТ
# ============================================================
print("\nРасчет корреляций с целевой переменной...")
corr_matrix = df_target.corr()
corr_series = corr_matrix[target_name].drop(target_name, errors='ignore').sort_values(ascending=False)

# Сохраняем в текстовый файл
txt_filename = f'full_correlations_{target_name.replace(" ", "_")}_all_features.txt'
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write(f"Корреляции с {target_name}\n")
    f.write(f"Количество пациентов: {len(df_target)}\n")
    f.write("-" * 80 + "\n")
    for feat, corr in corr_series.items():
        f.write(f"{feat:<50} {corr:>10.4f}\n")
print(f"Корреляции сохранены: {txt_filename}")

# Сохраняем в CSV
csv_filename = f'full_correlations_{target_name.replace(" ", "_")}_all_features.csv'
corr_series.to_csv(csv_filename, header=['Корреляция'], encoding='utf-8-sig')
print(f"Таблица корреляций сохранена: {csv_filename}")

print("\n" + "=" * 60)
print("ГОТОВО. Данные для модели самооценки здоровья подготовлены.")
print("=" * 60)