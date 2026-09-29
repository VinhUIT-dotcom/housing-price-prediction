import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Trích xuất và tạo các đặc trưng mới từ dữ liệu bất động sản
    """
    
    def __init__(self, df, output_dir='data/processed'):
        self.df = df.copy()
        self.output_dir = output_dir
        self.tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        os.makedirs(output_dir, exist_ok=True)
    
    def create_price_per_area(self):
        """
        Tạo đặc trưng: giá trên một đơn vị diện tích
        """
        logger.info("Tạo đặc trưng: Giá trên diện tích...")
        self.df['price_per_sqm'] = self.df['price'] / (self.df['area'] + 1e-6)
        return self.df
    
    def create_room_ratio(self):
        """
        Tạo đặc trưng: tỷ lệ phòng ngủ trên diện tích
        """
        logger.info("Tạo đặc trưng: Tỷ lệ phòng...")
        self.df['bedrooms_per_100sqm'] = (self.df['bedrooms'] / self.df['area']) * 100
        self.df['bathrooms_per_bedroom'] = self.df['bathrooms'] / (self.df['bedrooms'] + 1e-6)
        return self.df
    
    def create_age_feature(self):
        """
        Tạo đặc trưng: tuổi của tòa nhà
        """
        logger.info("Tạo đặc trưng: Tuổi tòa nhà...")
        from datetime import datetime
        current_year = datetime.now().year
        self.df['building_age'] = current_year - self.df['year_built']
        return self.df
    
    def create_frontage_ratio(self):
        """
        Tạo đặc trưng: tỷ lệ mặt tiền trên diện tích
        """
        logger.info("Tạo đặc trưng: Tỷ lệ mặt tiền...")
        self.df['frontage_area_ratio'] = self.df['street_frontage'] / (self.df['area'] + 1e-6)
        return self.df
    
    def extract_text_features(self):
        """
        Trích xuất đặc trưng từ dữ liệu text (description) sử dụng TF-IDF
        """
        logger.info("Trích xuất đặc trưng từ text (TF-IDF)...")
        
        if 'description' not in self.df.columns:
            logger.warning("Không có cột 'description', bỏ qua TF-IDF")
            return self.df
        
        # Trích xuất TF-IDF features
        tfidf_features = self.tfidf.fit_transform(self.df['description'].fillna(''))
        
        # Chuyển đổi thành DataFrame
        feature_names = [f'tfidf_{name}' for name in self.tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)
        
        # Kết hợp với dataframe gốc
        self.df = pd.concat([self.df.reset_index(drop=True), tfidf_df], axis=1)
        
        logger.info(f"✅ Đã tạo {len(feature_names)} TF-IDF features")
        return self.df
    
    def create_location_features(self):
        """
        Tạo đặc trưng dựa trên vị trí (district, ward)
        """
        logger.info("Tạo đặc trưng vị trí...")
        
        if 'district' in self.df.columns and 'ward' in self.df.columns:
            # Tạo combined location feature
            self.df['location'] = self.df['district'] + '_' + self.df['ward']
        
        return self.df
    
    def get_engineered_features(self):
        """
        Trả về tất cả các đặc trưng được tạo
        """
        return self.df
    
    def save_engineered_data(self, filename='engineered_housing_data.csv'):
        """
        Lưu dữ liệu với các đặc trưng được tạo
        """
        filepath = os.path.join(self.output_dir, filename)
        self.df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"✅ Đã lưu dữ liệu với đặc trưng vào: {filepath}")
        return filepath


def main():
    """
    Hàm main để chạy feature engineering
    """
    logger.info("="*50)
    logger.info("BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG")
    logger.info("="*50)
    
    # Tìm file dữ liệu đã xử lý
    processed_data_path = 'data/processed/processed_housing_data.csv'
    
    if not os.path.exists(processed_data_path):
        logger.error(f"Không tìm thấy file: {processed_data_path}")
        logger.info("Vui lòng chạy preprocessing.py trước!")
        return
    
    # Tải dữ liệu
    df = pd.read_csv(processed_data_path)
    logger.info(f"Đã tải {len(df)} bản ghi")
    
    # Tạo feature engineer
    engineer = FeatureEngineer(df)
    
    # Tạo các đặc trưng
    engineer.create_price_per_area()
    engineer.create_room_ratio()
    engineer.create_age_feature()
    engineer.create_frontage_ratio()
    engineer.create_location_features()
    engineer.extract_text_features()
    
    # Lưu dữ liệu
    engineer.save_engineered_data()
    
    # Hiển thị thông tin
    df_engineered = engineer.get_engineered_features()
    print("\n" + "="*50)
    print("THÔNG TIN CÁC ĐẶC TRƯNG MỚI")
    print("="*50)
    print(df_engineered.head())
    print(f"\nSố lượng đặc trưng: {df_engineered.shape[1]}")
    print(f"\nTên các đặc trưng:\n{list(df_engineered.columns)}")
    
    logger.info("✅ Hoàn thành trích xuất đặc trưng!")


if __name__ == "__main__":
    main()
