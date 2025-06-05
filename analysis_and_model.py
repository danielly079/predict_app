# import pandas as pd
# import numpy as np
# from ucimlrepo import fetch_ucirepo
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.metrics import (accuracy_score, confusion_matrix, 
#                            classification_report, roc_auc_score, roc_curve)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# def load_data():
#     """Загружает данные из UCI репозитория и выводит информацию о структуре"""
#     dataset = fetch_ucirepo(id=601)
    
#     # Объединяем признаки и целевую переменную
#     data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    
#     # Выводим информацию для отладки
#     print("="*50)
#     print("Загруженные столбцы:")
#     print(data.columns.tolist())
#     print("\nПервые 5 строк данных:")
#     print(data.head())
#     print("="*50)
    
#     return data

# def preprocess_data(data):
#     """Предварительная обработка данных"""
#     # Проверяем, какие столбцы действительно существуют
#     existing_cols = data.columns.tolist()
#     print("Существующие столбцы:", existing_cols)
    
#     # Удаляем только те столбцы, которые есть в данных
#     cols_to_drop = []
#     for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
#         if col in existing_cols:
#             cols_to_drop.append(col)
    
#     if cols_to_drop:
#         data = data.drop(columns=cols_to_drop)
    
#     # Кодируем категориальный признак 'Type'
#     if 'Type' in data.columns:
#         data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})
#     else:
#         raise KeyError("Столбец 'Type' не найден в данных")
    
#     # Определяем числовые столбцы для масштабирования (используем фактические названия из вашего датасета)
#     numerical_cols = []
#     for col in ['Air temperature', 'Process temperature', 
#                'Rotational speed', 'Torque', 'Tool wear']:
#         if col in data.columns:
#             numerical_cols.append(col)
#         else:
#             print(f"Предупреждение: столбец {col} не найден в данных")
    
#     if not numerical_cols:
#         raise KeyError("Не найдены числовые столбцы для масштабирования")
    
#     print("Столбцы для масштабирования:", numerical_cols)
    
#     # Масштабируем числовые признаки
#     scaler = StandardScaler()
#     data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
#     return data, scaler, numerical_cols
    
    
# def train_models(X_train, y_train):
#     """Обучение моделей"""
#     models = {
#         'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#         'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#     }
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         print(f"Модель {name} обучена")
    
#     return models

# def evaluate_models(models, X_test, y_test):
#     """Оценка качества моделей"""
#     results = {}
#     plt.figure(figsize=(10, 6))
    
#     for name, model in models.items():
#         y_pred = model.predict(X_test)
#         y_pred_proba = model.predict_proba(X_test)[:, 1]
        
#         # Рассчитываем метрики
#         accuracy = accuracy_score(y_test, y_pred)
#         roc_auc = roc_auc_score(y_test, y_pred_proba)
#         cm = confusion_matrix(y_test, y_pred)
#         report = classification_report(y_test, y_pred)
        
#         # ROC-кривая
#         fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#         plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
#         results[name] = {
#             'accuracy': accuracy,
#             'roc_auc': roc_auc,
#             'confusion_matrix': cm,
#             'classification_report': report
#         }
    
#     # Настраиваем график ROC-кривых
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC-кривые моделей')
#     plt.legend()
#     plt.close()
    
#     return results, plt.gcf()

# def run_analysis():
#     """Основной анализ"""
#     try:
#         # 1. Загрузка данных
#         print("Загрузка данных...")
#         data = load_data()
        
#         # 2. Предобработка
#         print("Предобработка данных...")
#         data, scaler, numerical_cols = preprocess_data(data)
#         print("Используемые числовые столбцы:", numerical_cols)
        
#         # 3. Разделение данных
#         X = data.drop(columns=['Machine failure'])
#         y = data['Machine failure']
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42)
        
#         # 4. Обучение моделей
#         print("Обучение моделей...")
#         models = train_models(X_train, y_train)
        
#         # 5. Оценка моделей
#         print("Оценка моделей...")
#         results, roc_curve = evaluate_models(models, X_test, y_test)
        
#         return {
#             'data': data,
#             'models': models,
#             'results': results,
#             'roc_curve': roc_curve,
#             'scaler': scaler,
#             'numerical_cols': numerical_cols
#         }
    
#     except Exception as e:
#         print(f"Ошибка в run_analysis: {str(e)}")
#         raise

# def predict_new_data(model, scaler, numerical_cols, input_data):
#     """Предсказание для новых данных"""



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo 

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    try:
        dataset = fetch_ucirepo(id=601)
        
        if dataset is not None:
            # Объединение признаков и целевых переменных
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            
            # Предобработка данных
            st.header("Предобработка данных")
            
            # Удаление ненужных столбцов
            columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            columns_to_drop = [col for col in columns_to_drop if col in data.columns]
            data = data.drop(columns=columns_to_drop)
            
            # Преобразование категориальной переменной Type
            le = LabelEncoder()
            data['Type'] = le.fit_transform(data['Type'])
            
            # Проверка на пропущенные значения
            st.subheader("Пропущенные значения")
            st.write(data.isnull().sum())
            
            if data.isnull().sum().sum() > 0:
                st.warning("Обнаружены пропущенные значения. Они будут заполнены медианными значениями.")
                data = data.fillna(data.median())
            
            # Масштабирование числовых признаков
            numerical_features = ['Air temperature', 'Process temperature', 
                                'Rotational speed', 'Torque', 'Tool wear']
            scaler = StandardScaler()
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            
            # Разделение данных
            st.header("Разделение данных")
            X = data.drop(columns=['Machine failure'])
            y = data['Machine failure']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.write(f"Обучающая выборка: {X_train.shape[0]} записей")
            st.write(f"Тестовая выборка: {X_test.shape[0]} записей")
            
            # Обучение моделей
            st.header("Обучение моделей")
            
            # Logistic Regression
            st.subheader("Logistic Regression")
            log_reg = LogisticRegression(random_state=42, max_iter=1000)
            log_reg.fit(X_train, y_train)
            
            # Random Forest
            st.subheader("Random Forest")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # XGBoost
            st.subheader("XGBoost")
            xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb.fit(X_train, y_train)
            
            # Функция для оценки модели
            def evaluate_model(model, model_name, X_test, y_test):
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                st.write(f"Accuracy: {accuracy:.4f}")
                st.write(f"ROC-AUC: {roc_auc:.4f}")
                
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Нет отказа', 'Отказ'],
                           yticklabels=['Нет отказа', 'Отказ'])
                plt.xlabel('Предсказание')
                plt.ylabel('Фактическое значение')
                st.pyplot(fig)
                
                st.subheader("Classification Report")
                st.text(class_report)
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                st.pyplot(plt)
                plt.close('all')
            
            # Оценка моделей
            st.header("Оценка моделей")
            
            st.subheader("Logistic Regression")
            evaluate_model(log_reg, "Logistic Regression", X_test, y_test)
            
            st.subheader("Random Forest")
            evaluate_model(rf, "Random Forest", X_test, y_test)
            
            st.subheader("XGBoost")
            evaluate_model(xgb, "XGBoost", X_test, y_test)
            
            # Интерфейс для предсказания
            st.header("Предсказание по новым данным")
            with st.form("prediction_form"):
                st.write("Введите значения признаков для предсказания:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
                    air_temp = st.number_input("Температура окружающей среды [K]", min_value=0.0, value=300.0)
                    process_temp = st.number_input("Рабочая температура [K]", min_value=0.0, value=310.0)
                    
                with col2:
                    rotational_speed = st.number_input("Скорость вращения [rpm]", min_value=0, value=1500)
                    torque = st.number_input("Крутящий момент [Nm]", min_value=0.0, value=40.0)
                    tool_wear = st.number_input("Износ инструмента [min]", min_value=0, value=0)
                
                submit_button = st.form_submit_button("Предсказать")
                
                if submit_button:
                    try:
                        # Преобразование введенных данных
                        input_data = pd.DataFrame({
                            'Type': [product_type],
                            'Air temperature': [air_temp],
                            'Process temperature': [process_temp],
                            'Rotational speed': [rotational_speed],
                            'Torque': [torque],
                            'Tool wear': [tool_wear]
                        })
                        
                        # Преобразование категориальной переменной
                        input_data['Type'] = le.transform(input_data['Type'])
                        
                        # Масштабирование числовых признаков
                        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                        
                        # Предсказание для каждой модели
                        st.subheader("Результаты предсказания")
                        
                        models = {
                            "Logistic Regression": log_reg,
                            "Random Forest": rf,
                            "XGBoost": xgb
                        }
                        
                        results = []
                        for model_name, model in models.items():
                            prediction = model.predict(input_data)[0]
                            prediction_proba = model.predict_proba(input_data)[0, 1]
                            results.append({
                                "Модель": model_name,
                                "Предсказание": "Отказ" if prediction == 1 else "Нет отказа",
                                "Вероятность отказа": f"{prediction_proba:.4f}"
                            })
                        
                        # Отображение результатов в таблице
                        st.table(pd.DataFrame(results))
                        
                    except Exception as e:
                        st.error(f"Ошибка при выполнении предсказания: {str(e)}")
        else:
            st.error("Не удалось загрузить данные. Пожалуйста, попробуйте позже.")
    
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке данных: {str(e)}")
#analysis_and_model_page()