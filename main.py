from data_preparation import load_and_split_data
from model_training import build_and_train_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def main():
    # Путь к датасету
    file_path = '/content/drive/MyDrive/Laptop_price.csv'
    
    # Путь для сохранения модели
    model_save_path = '/content/drive/MyDrive/laptop_price_model.pkl'
    
    # Загружаем и разделяем данные
    X_train, X_test, y_train, y_test, _ = load_and_split_data(file_path)
    
    # Строим и обучаем пайплайн
    pipeline = build_and_train_pipeline(X_train, y_train, model_save_path)
    
    # Оцениваем модель
    y_pred = pipeline.predict(X_test)
    
    # Рассчитываем метрики
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Средняя абсолютная ошибка: {mae:.2f}")
    print(f"Корень среднеквадратичной ошибки: {rmse:.2f}")
    print(f"Коэффициент детерминации R²: {r2:.2f}")

if __name__ == "__main__":
    main()
