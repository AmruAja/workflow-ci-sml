# 1. Gunakan base image Python versi 3.9 yang ringan
FROM python:3.9-slim

# 2. Tentukan folder kerja di dalam container nanti
WORKDIR /app

# 3. Salin semua file dari folder MLProject laptop/GitHub kamu ke dalam container
COPY . /app

# 4. Upgrade pip dan install semua library yang tertulis di ci.yaml kamu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mlflow pandas numpy scikit-learn

# 5. Jalankan perintah utama untuk mengeksekusi script modeling kamu saat container menyala
CMD ["python", "modelling.py"]
