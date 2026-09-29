import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Xử lý và làm sạch dữ liệu bất động sản
    """
    
    def __init__(self, input_path, output_dir='data/processed'):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """
        Tải dữ liệu từ file CSV
        """
        logger.info(f"Đang tải dữ liệu từ: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        logger.info(f"✅ Đã tải {len(self.df)} bản ghi")
        return self.df
    
    def handle_missing_values(self, strategy='mean'):
        """
        Xử lý dữ liệu thiếu
        
        Args:
            strategy: 'mean', 'median', hoặc 'drop'
        """
        logger.info("Xử lý dữ liệu thiếu...")
        
        missing_count = self.df.isnull().sum()
        if missing_count.sum() > 0:
            logger.info(f"Dữ liệu thiếu:\n{missing_count[missing_count > 0]}")
            
            if strategy == 'mean':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            elif strategy == 'drop':
                self.df = self.df.dropna()
        
        logger.info("✅ Xử lý dữ liệu thiếu hoàn tất")
        return self.df
    
    def remove_outliers(self, columns=None, threshold=3):
        """
        Loại bỏ outliers sử dụng Z-score
        
        Args:
            columns: Danh sách cột để kiểm tra
            threshold: Ngưỡng Z-score (mặc định 3)
        """
        logger.info("Loại bỏ outliers...")
        
        if columns is None:
            columns = ['price', 'area', 'bedrooms', 'floor']
        
        initial_len = len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
        
        removed = initial_len - len(self.df)
        logger.info(f"✅ Đã loại bỏ {removed} bản ghi (outliers)")
        return self.df
    
    def normalize_features(self, columns=None):
        """
        Chuẩn hóa các đặc trưng số
        
        Args:
            columns: Danh sách cột để chuẩn hóa
        """
        logger.info("Chuẩn hóa các đặc trưng...")
        
        if columns is None:
            columns = ['price', 'area', 'bedrooms', 'bathrooms', 'floor', 'street_frontage']
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.scaler.fit_transform(self.df[[col]])
        
        logger.info("✅ Chuẩn hóa hoàn tất")
        return self.df
    
    def encode_categorical(self, columns=None):
        """
        Mã hóa các cột phân loại
        
        Args:
            columns: Danh sách cột phân loại
        """
        logger.info("Mã hóa dữ liệu phân loại...")
        
        if columns is None:
            columns = ['district', 'ward']
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        
        logger.info("✅ Mã hóa hoàn tất")
        return self.df
    
    def save_processed_data(self, filename='processed_housing_data.csv'):
        """
        Lưu dữ liệu đã xử lý
        """
        filepath = os.path.join(self.output_dir, filename)
        self.df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"✅ Đã lưu dữ liệu xử lý vào: {filepath}")
        return filepath
    
    def get_dataframe(self):
        """
        Trả về DataFrame đã xử lý
        """
        return self.df


def main():
    """
    Hàm main để chạy preprocessing
    """
    logger.info("="*50)
    logger.info("BẮT ĐẦU XỬ LÝ DỮ LIỆU")
    logger.info("="*50)
    
    # Tìm file dữ liệu thô
    raw_data_path = 'data/raw/housing_data.csv'
    
    if not os.path.exists(raw_data_path):
        logger.error(f"Không tìm thấy file: {raw_data_path}")
        logger.info("Vui lòng chạy scraper.py trước!")
        return
    
    # Tạo preprocessor
    preprocessor = DataPreprocessor(raw_data_path)
    
    # Xử lý dữ liệu
    preprocessor.load_data()
    preprocessor.handle_missing_values(strategy='mean')
    preprocessor.remove_outliers()
    preprocessor.encode_categorical()
    preprocessor.normalize_features()
    
    # Lưu dữ liệu
    preprocessor.save_processed_data()
    
    # Hiển thị thông tin
    df = preprocessor.get_dataframe()
    print("\n" + "="*50)
    print("THÔNG TIN DỮ LIỆU SAU XỬ LÝ")
    print("="*50)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nThống kê mô tả:\n{df.describe()}")
    
    logger.info("✅ Hoàn thành xử lý dữ liệu!")


if __name__ == "__main__":
    main()
