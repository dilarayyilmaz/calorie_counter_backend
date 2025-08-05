import os
import io
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from PIL import Image
from thefuzz import process  # <-- YENİ İMPORT

# --- 1. UYGULAMA VE MODEL KURULUMU ---

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- MODELİ YÜKLE ---
app.logger.info("TensorFlow ve Keras Modeli Yükleniyor...")
MODEL_PATH = "food_model_mobilenet.h5"
model = load_model(MODEL_PATH)
app.logger.info("Model Başarıyla Yüklendi.")

# --- BESİN DEĞERİ VERİTABANINI HAZIRLA ---
app.logger.info("Besin Değeri Veritabanı Hazırlanıyor...")
CSV_PATH = 'food101_with_nutrition_CLEAN.csv'
df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()
df['class_name'] = df['class_name'].str.strip().str.lower()
df.dropna(subset=['calories'], inplace=True)
df.drop_duplicates(subset=['class_name'], keep='first', inplace=True)

# ÖNEMLİ: Bulanık arama için index yerine sütunu kullanacağız.
# Bu yüzden set_index işlemini şimdilik kaldırıyoruz veya bir kopyasını tutuyoruz.
nutrition_db_df = df.copy()
# Arama için tüm olası yemek isimlerini bir listeye alalım.
db_class_names = nutrition_db_df['class_name'].tolist()

# Orijinal index'li veritabanını da tutabiliriz.
nutrition_db_indexed = df.set_index('class_name')

app.logger.info("Besin Değeri Veritabanı Hazırlandı.")
app.logger.info(f"Veritabanında {len(nutrition_db_df)} adet benzersiz yemek kaydı bulundu.")

# --- SINIF İSİMLERİNİ YÜKLE ---
try:
    with open('class_indices.json', 'r') as f:
        class_names = json.load(f)['class_names']
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    app.logger.info("Sınıf İndeksleri 'class_indices.json' dosyasından başarıyla yüklendi.")
except FileNotFoundError:
    app.logger.warning("[UYARI] 'class_indices.json' bulunamadı. Bu, yanlış tahminlere neden olabilir!")
    unique_classes_from_csv = sorted(df['class_name'].unique())
    idx_to_class = {i: name.replace(' ', '_') for i, name in enumerate(unique_classes_from_csv)}


def prepare_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)


# --- 3. API ENDPOINT'İ (Bulanık Arama ile Güncellendi) ---
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Lütfen 'image' anahtarıyla bir resim dosyası gönderin."}), 400

    file = request.files['image']
    if not file or not file.filename:
        return jsonify({"error": "Lütfen bir dosya seçin."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Geçersiz dosya formatı. İzin verilen formatlar: {list(ALLOWED_EXTENSIONS)}"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        prepared_image = prepare_image(img)

        app.logger.info("Model tahmini yapılıyor...")
        predictions = model.predict(prepared_image)
        app.logger.info("Tahmin başarıyla yapıldı.")

        class_probabilities = predictions[0][0]
        predicted_index = np.argmax(class_probabilities)
        confidence = float(class_probabilities[predicted_index])
        predicted_calorie = predictions[1][0][0]

        class_name_from_model = idx_to_class.get(predicted_index, "unknown")
        class_name_for_db = class_name_from_model.replace('_', ' ').lower()

        # === BULANIK EŞLEŞTİRME (FUZZY MATCHING) BÖLÜMÜ ===

        app.logger.info(f"Modelden gelen ham isim: '{class_name_for_db}'. Veritabanında en yakın eşleşme aranıyor...")

        # `thefuzz.process.extractOne` fonksiyonu, bir string'e bir liste içindeki en çok benzeyen elemanı bulur.
        # Geriye (bulunan_eleman, benzerlik_skoru) şeklinde bir tuple döndürür.
        # score_cutoff: Belirli bir benzerlik skorunun altındaki eşleşmeleri kabul etme. (0-100 arası)
        best_match = process.extractOne(class_name_for_db, db_class_names, score_cutoff=75)

        predicted_protein = 0.0
        predicted_fats = 0.0
        predicted_carbs = 0.0
        readable_class_name = class_name_for_db.title()  # Varsayılan isim

        if best_match:
            # Eğer yeterince iyi bir eşleşme bulunduysa
            matched_name, match_score = best_match
            app.logger.info(f"En iyi eşleşme bulundu: '{matched_name}' (Benzerlik: {match_score}%)")

            # Bulunan isimle veritabanından bilgileri çek.
            # Burada index'li veritabanını kullanmak daha hızlıdır.
            food_info = nutrition_db_indexed.loc[matched_name]

            # Kullanıcıya gösterilecek ismi, veritabanında bulduğumuz isimle güncelleyelim.
            readable_class_name = matched_name.title()

            try:
                predicted_protein = float(food_info['protein'])
                predicted_fats = float(food_info['fats'])
                predicted_carbs = float(food_info['carbohydrates'])
            except (ValueError, TypeError) as e:
                app.logger.warning(
                    f"DEĞER HATASI: '{matched_name}' için besin değerleri sayıya dönüştürülemedi. Hata: {e}")

        else:
            # Yeterince iyi bir eşleşme bulunamadıysa
            app.logger.warning(
                f"EŞLEŞME BULUNAMADI: '{class_name_for_db}' için veritabanında yeterli benzerlikte (%75+) bir kayıt bulunamadı.")
            # Bu durumda makrolar 0 kalacak.

        # Flutter uygulamasına gönderilecek sonucu oluştur
        result = {
            "food_name": readable_class_name,
            "confidence": round(confidence * 100, 2),
            "calories": int(round(predicted_calorie, 0)),
            "protein": round(predicted_protein, 1),
            "fat": round(predicted_fats, 1),
            "carbs": round(predicted_carbs, 1)
        }

        app.logger.info(f"Sonuç başarıyla oluşturuldu: {json.dumps(result, indent=2)}")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Tahmin sürecinde beklenmedik bir hata oluştu: {e}", exc_info=True)
        return jsonify({"error": f"Resim işlenirken bir hata oluştu: {str(e)}"}), 500


# --- Sunucuyu Başlatma Bloğu ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)