# 🏠 Dự Báo Giá Nhà - Housing Price Prediction

Dự án Machine Learning để dự báo giá bất động sản sử dụng các mô hình học máy khác nhau.

## 📋 Mục Tiêu Dự Án

- Thu thập dữ liệu từ các trang rao vặt bất động sản (≥5000 mẫu)
- Xử lý và làm sạch dữ liệu
- Xây dựng các mô hình dự báo giá
- Đánh giá và so sánh hiệu năng các mô hình

## 🔧 Yêu Cầu Kỹ Thuật

### Dữ Liệu
- Tối thiểu 5,000 mẫu tin
- Các đặc trưng: địa chỉ, số phòng ngủ, số tầng, chiều rộng mặt tiền, giá tiền, v.v.
- Xử lý outliers và dữ liệu thiếu
- Chuẩn hóa địa chỉ theo hành chính (Quận/Huyện, Phường/Xã)

### Kỹ Thuật
- **Feature Engineering**: TF-IDF, Word Embedding cho dữ liệu text
- **Mô hình**:
  - Linear Regression
  - Random Forest
  - XGBoost
  - LightGBM
- **Đánh giá**: RMSE, MAE, R² Score

## 📁 Cấu Trúc Thư Mục

```
housing-price-prediction/
├── data/
│   ├── raw/                    # Dữ liệu thô
│   ├── processed/              # Dữ liệu đã xử lý
│   └── datasets.py             # Script tải dữ liệu
├── src/
│   ├── scraper.py             # Thu thập dữ liệu (web scraping)
│   ├── preprocessing.py        # Xử lý dữ liệu
│   ├── feature_engineering.py  # Kỹ thuật trích xuất đặc trưng
│   └── models.py              # Xây dựng các mô hình
├── notebooks/
│   └── exploration.ipynb      # EDA (Exploratory Data Analysis)
├── results/
│   └── evaluation.csv         # Kết quả đánh giá mô hình
├── requirements.txt           # Các thư viện cần thiết
├── .gitignore
└── README.md
```

## 🚀 Bắt Đầu

### 1. Clone Repository
```bash
git clone https://github.com/VinhUIT-dotcom/housing-price-prediction.git
cd housing-price-prediction
```

### 2. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 3. Thu thập dữ liệu
```bash
python src/scraper.py
```

### 4. Xử lý dữ liệu
```bash
python src/preprocessing.py
```

### 5. Xây dựng mô hình
```bash
python src/models.py
```

## 📊 Kết Quả

Các mô hình sẽ được đánh giá dựa trên:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** Score

## 📝 License

MIT License

---

**Tác giả**: VinhUIT-dotcom
**Ngày tạo**: 2026
