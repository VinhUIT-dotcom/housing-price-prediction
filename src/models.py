import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import logging
import os
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HousingPricePredictor:
    """
    Xây dựng và đánh giá các mô hình dự báo giá nhà
    """
    
    def __init__(self, data_path, output_dir='results', test_size=0.2, random_state=42):
        self.data_path = data_path
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        os.makedirs(output_dir, exist_ok=True)
    
    def load_and_prepare_data(self, target_column='price'):
        """
        Tải và chuẩn bị dữ liệu cho huấn luyện
        """
        logger.info("Tải và chuẩn bị dữ liệu...")
        
        df = pd.read_csv(self.data_path)
        
        # Chọn các đặc trưng số để huấn luyện
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column not in numeric_cols:
            logger.error(f"Cột mục tiêu '{target_column}' không phải là số!")
            return
        
        # Tách đặc trưng và mục tiêu
        X = df[numeric_cols].drop(columns=[target_column])
        y = df[target_column]
        
        # Xóa các hàng có giá trị NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Tổng số mẫu: {len(X)}")
        logger.info(f"Số lượng đặc trưng: {X.shape[1]}")
        
        # Chia thành train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"✅ Train set: {len(self.X_train)}, Test set: {len(self.X_test)}")
        return X, y
    
    def train_linear_regression(self):
        """
        Huấn luyện mô hình Linear Regression
        """
        logger.info("\nHuấn luyện Linear Regression...")
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = model
        logger.info("✅ Hoàn tất")
        return model
    
    def train_random_forest(self, n_estimators=100, random_state=42):
        """
        Huấn luyện mô hình Random Forest
        """
        logger.info("\nHuấn luyện Random Forest...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        logger.info("✅ Hoàn tất")
        return model
    
    def train_xgboost(self, n_estimators=100, random_state=42):
        """
        Huấn luyện mô hình XGBoost
        """
        logger.info("\nHuấn luyện XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbosity=0
        )
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        logger.info("✅ Hoàn tất")
        return model
    
    def train_lightgbm(self, n_estimators=100, random_state=42):
        """
        Huấn luyện mô hình LightGBM
        """
        logger.info("\nHuấn luyện LightGBM...")
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['LightGBM'] = model
        logger.info("✅ Hoàn tất")
        return model
    
    def evaluate_model(self, model_name, model):
        """
        Đánh giá mô hình sử dụng các chỉ số RMSE, MAE, R²
        """
        # Dự báo trên test set
        y_pred = model.predict(self.X_test)
        
        # Tính các chỉ số
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Lưu kết quả
        result = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.results.append(result)
        
        logger.info(f"{model_name}:")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  R² Score: {r2:.4f}")
        
        return result
    
    def evaluate_all_models(self):
        """
        Đánh giá tất cả các mô hình đã huấn luyện
        """
        logger.info("\n" + "="*50)
        logger.info("ĐÁNH GIÁ CÁC MÔ HÌNH")
        logger.info("="*50)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)
    
    def save_results(self, filename='model_evaluation.csv'):
        """
        Lưu kết quả đánh giá vào file CSV
        """
        results_df = pd.DataFrame(self.results)
        filepath = os.path.join(self.output_dir, filename)
        results_df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"\n✅ Đã lưu kết quả vào: {filepath}")
        
        print("\n" + "="*50)
        print("KẾT QUẢ ĐÁNH GIÁ")
        print("="*50)
        print(results_df.to_string(index=False))
        
        return filepath
    
    def save_models(self):
        """
        Lưu các mô hình đã huấn luyện
        """
        models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"✅ Đã lưu mô hình: {model_path}")


def main():
    """
    Hàm main để chạy huấn luyện và đánh giá mô hình
    """
    logger.info("="*50)
    logger.info("HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH")
    logger.info("="*50)
    
    # Tìm file dữ liệu với các đặc trưng
    data_path = 'data/processed/engineered_housing_data.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Không tìm thấy file: {data_path}")
        logger.info("Vui lòng chạy feature_engineering.py trước!")
        return
    
    # Tạo predictor
    predictor = HousingPricePredictor(data_path)
    
    # Tải và chuẩn bị dữ liệu
    predictor.load_and_prepare_data(target_column='price')
    
    # Huấn luyện các mô hình
    predictor.train_linear_regression()
    predictor.train_random_forest(n_estimators=100)
    predictor.train_xgboost(n_estimators=100)
    predictor.train_lightgbm(n_estimators=100)
    
    # Đánh giá các mô hình
    predictor.evaluate_all_models()
    
    # Lưu kết quả và mô hình
    predictor.save_results()
    predictor.save_models()
    
    logger.info("\n✅ Hoàn thành huấn luyện và đánh giá!")


if __name__ == "__main__":
    main()
