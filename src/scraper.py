import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import logging
import os

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealEstateScraper:
    """
    Web Scraper để thu thập dữ liệu bất động sản từ các trang rao vặt
    """
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        self.data = []
        os.makedirs(output_dir, exist_ok=True)
    
    def scrape_sample_data(self, num_samples=100):
        """
        Tạo dữ liệu mẫu cho demo (thay thế cho web scraping thực tế)
        Trong thực tế, bạn sẽ thay thế bằng code scraping từ các website thực tế
        
        Args:
            num_samples: Số lượng mẫu dữ liệu cần tạo
        """
        import random
        import numpy as np
        
        logger.info(f"Tạo {num_samples} mẫu dữ liệu mẫu...")
        
        districts = ['Quận 1', 'Quận 2', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 7', 
                     'Quận 10', 'Bình Tân', 'Bình Thạnh', 'Tân Bình']
        wards = ['Phường Bến Nghé', 'Phường Đa Kao', 'Phường Tân Định',
                 'Phường Võ Thị Sáu', 'Phường 1', 'Phường 2']
        
        for i in range(num_samples):
            price = np.random.normal(loc=3000, scale=1000)  # Giá trung bình 3 tỷ
            area = np.random.normal(loc=100, scale=30)  # Diện tích trung bình 100m2
            bedrooms = random.randint(1, 5)  # 1-5 phòng ngủ
            bathrooms = random.randint(1, 4)  # 1-4 phòng tắm
            floor = random.randint(1, 10)  # 1-10 tầng
            street_frontage = np.random.uniform(3, 20)  # 3-20m mặt tiền
            year_built = random.randint(1995, 2023)  # Năm xây dựng
            
            self.data.append({
                'id': i + 1,
                'price': max(price, 500),  # Giá tối thiểu 500 triệu
                'area': max(area, 20),  # Diện tích tối thiểu 20m2
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'floor': floor,
                'street_frontage': street_frontage,
                'year_built': year_built,
                'district': random.choice(districts),
                'ward': random.choice(wards),
                'description': f'Nhà {bedrooms}PN, {bathrooms}WC, {area:.0f}m2, mặt tiền {street_frontage:.1f}m',
                'scraped_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        logger.info(f"✅ Đã tạo {len(self.data)} mẫu dữ liệu")
        return self.data
    
    def save_to_csv(self, filename='housing_data.csv'):
        """
        Lưu dữ liệu vào file CSV
        """
        if not self.data:
            logger.warning("Không có dữ liệu để lưu!")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"✅ Đã lưu dữ liệu vào: {filepath}")
        logger.info(f"Tổng số bản ghi: {len(df)}")
        return filepath
    
    def get_dataframe(self):
        """
        Trả về DataFrame từ dữ liệu đã scrape
        """
        return pd.DataFrame(self.data)


def main():
    """
    Hàm main để chạy scraper
    """
    logger.info("=" * 50)
    logger.info("BẮT ĐẦU THU THẬP DỮ LIỆU BẤT ĐỘNG SẢN")
    logger.info("=" * 50)
    
    # Tạo scraper
    scraper = RealEstateScraper(output_dir='data/raw')
    
    # Tạo dữ liệu mẫu (100 mẫu cho demo, trong thực tế là 5000+)
    scraper.scrape_sample_data(num_samples=100)
    
    # Lưu dữ liệu
    scraper.save_to_csv()
    
    # Hiển thị thông tin
    df = scraper.get_dataframe()
    print("\n" + "="*50)
    print("THÔNG TIN DỮ LIỆU")
    print("="*50)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nThông tin cột:\n{df.info()}")
    
    logger.info("✅ Hoàn thành thu thập dữ liệu!")


if __name__ == "__main__":
    main()
