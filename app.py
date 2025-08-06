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
from thefuzz import process
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. UYGULAMA VE LOGLAMA KURULUMU ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- GEMINI CHATBOT KURULUMU ---
load_dotenv()
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        app.logger.warning("[UYARI] GEMINI_API_KEY .env dosyasında bulunamadı. Chatbot çalışmayabilir.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    app.logger.info("Google Gemini Modeli başarıyla yapılandırıldı.")
except Exception as e:
    app.logger.error(f"Gemini modeli başlatılamadı: {e}")
    gemini_model = None

# --- DOSYA YOLLARI VE YÜKLEME ---
MODEL_PATH = "food_model_mobilenet.h5"
NUTRITION_CSV_PATH = "food101_with_nutrition_CLEAN.csv"

try:
    app.logger.info(f"'{MODEL_PATH}' modeli yükleniyor...")
    model = load_model(MODEL_PATH)
    app.logger.info("✅ Model başarıyla yüklendi.")

    app.logger.info(f"'{NUTRITION_CSV_PATH}' veritabanı hazırlanıyor...")
    df = pd.read_csv(NUTRITION_CSV_PATH)
    df.columns = df.columns.str.strip()
    df['class_name'] = df['class_name'].str.strip().str.lower()
    df.dropna(subset=['calories'], inplace=True)
    df.drop_duplicates(subset=['class_name'], keep='first', inplace=True)

    db_class_names = df['class_name'].tolist()
    nutrition_db_indexed = df.set_index('class_name')
    app.logger.info("✅ Besin değeri veritabanı hazırlandı.")

except FileNotFoundError as e:
    app.logger.error(f"HATA: Dosya bulunamadı! Lütfen dosya adlarını kontrol et. Detay: {e}")
    model = None
    db_class_names = []
    nutrition_db_indexed = None
except Exception as e:
    app.logger.error(f"Model veya veri yüklenirken kritik bir hata oluştu: {e}")


# --- YARDIMCI FONKSİYONLAR ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def prepare_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)


# --- /predict ENDPOINT'İ (Yemek Tahmini) ---
@app.route("/predict", methods=["POST"])
def predict():
    if not model or nutrition_db_indexed is None:
        return jsonify({"error": "Sunucu başlangıçta model veya veritabanı dosyalarını yükleyemedi."}), 503

    if 'image' not in request.files:
        return jsonify({"error": "Lütfen 'image' anahtarıyla bir resim dosyası gönderin."}), 400
    file = request.files['image']
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Geçersiz dosya."}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        prepared_image = prepare_image(img)
        predictions = model.predict(prepared_image)
        class_probabilities = predictions[0][0]
        predicted_index = np.argmax(class_probabilities)
        confidence = float(class_probabilities[predicted_index])
        predicted_calorie = predictions[1][0][0]

        try:
            model_class_labels = sorted(db_class_names)
            class_name_from_model = model_class_labels[predicted_index]
        except IndexError:
            class_name_from_model = "unknown"

        name_to_search = class_name_from_model.replace('_', ' ').lower()
        best_match = process.extractOne(name_to_search, db_class_names, score_cutoff=80)

        predicted_protein, predicted_fats, predicted_carbs = 0.0, 0.0, 0.0
        readable_class_name = name_to_search.title()

        if best_match:
            matched_name, score = best_match
            readable_class_name = matched_name.title()
            food_info = nutrition_db_indexed.loc[matched_name]
            predicted_protein = float(food_info.get('protein', 0))
            predicted_fats = float(food_info.get('fats', 0))
            predicted_carbs = float(food_info.get('carbohydrates', 0))

        result = {
            "food_name": readable_class_name,
            "confidence": round(confidence * 100, 2),
            "calories": int(round(predicted_calorie, 0)),
            "protein": round(predicted_protein, 1),
            "fat": round(predicted_fats, 1),
            "carbs": round(predicted_carbs, 1)
        }
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Tahmin sürecinde hata: {e}", exc_info=True)
        return jsonify({"error": f"Resim işlenirken bir hata oluştu: {str(e)}"}), 500


# --- /chat ENDPOINT'İ (Beslenme Danışmanı) ---
@app.route("/chat", methods=["POST"])
def chat_with_advisor():
    if not gemini_model:
        return jsonify({"error": "Danışman (Gemini) modeli başlatılamadı."}), 503
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Lütfen 'question' anahtarıyla bir soru gönderin."}), 400
    user_question = data['question']
    meal_history = data.get('history', [])
    prompt_template = """Sen, kullanıcıların daha sağlıklı seçimler yapmasına yardımcı olan, arkadaş canlısı ve bilgili bir beslenme danışmanı botusun. Cevapların her zaman pozitif, teşvik edici ve yargılayıcı olmamalı. Kullanıcının yemek geçmişini ve mevcut sorusunu analiz ederek, basit ve uygulanabilir tavsiyeler ver. Kısa, net ve anlaşılır konuş. Her zaman Türkçe yanıt ver.
---
KULLANICI BİLGİLERİ:
{history}
KULLANICININ SORUSU:
"{question}"
SENİN CEVABIN:
"""
    history_text = "Kullanıcının son yediklerinin özeti:\n"
    if meal_history:
        for meal in meal_history:
            history_text += f"- {meal.get('food_name', '?')} ({meal.get('calories', 0)} kcal)\n"
    else:
        history_text += "- Kullanıcının yemek geçmişi henüz boş.\n"
    final_prompt = prompt_template.format(history=history_text, question=user_question)
    try:
        response = gemini_model.generate_content(final_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        app.logger.error(f"Gemini Chatbot hatası: {e}", exc_info=True)
        return jsonify({"error": f"Danışmanla konuşurken bir hata oluştu: {str(e)}"}), 500


# --- /recipe_chat ENDPOINT'İ (Yemek Tarifi Asistanı) ---
@app.route("/recipe_chat", methods=["POST"])
def recipe_chat_with_advisor():
    if not gemini_model:
        return jsonify({"error": "Gemini modeli başlatılamadı."}), 503
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Lütfen 'question' anahtarıyla bir soru gönderin."}), 400
    user_question = data['question']

    prompt_template = """Sen, sağlıklı, pratik ve lezzetli yemek tarifleri konusunda uzman, yaratıcı bir mutfak şefisin. Kullanıcının isteğine göre tarifler sun. Cevaplarını SADECE ve SADECE aşağıdaki formatta ver. Başka hiçbir metin veya açıklama ekleme. Etiketleri (###...###) asla değiştirme ve her zaman aynı şekilde kullan.

###BAŞLIK###
Tarifin Adı
###MALZEMELER###
- Malzeme 1
- Malzeme 2
###YAPILIŞI###
1. Adım...
2. Adım...

KULLANICININ İSTEĞİ:
"{question}"

SENİN TARİFİN:
"""
    final_prompt = prompt_template.format(question=user_question)
    try:
        response = gemini_model.generate_content(final_prompt)
        recipe_text = response.text

        # --- YENİ EKLENEN KISIM: Gelen metni doğru formata getiriyoruz ---
        lines = recipe_text.strip().split('\n')
        if lines and lines[0].startswith('###') and lines[0].endswith('###'):
            lines[0] = '###BAŞLIK###'
            recipe_text = '\n'.join(lines)
        # --------------------------------------------------------------

        return jsonify({"answer": recipe_text})
    except Exception as e:
        app.logger.error(f"Gemini Tarif Chatbot hatası: {e}", exc_info=True)
        return jsonify({"error": f"Tarif alınırken bir hata oluştu: {str(e)}"}), 500


# --- /get_macros ENDPOINT'İ (Yemek Adından Makro Bilgisi Çekme) ---
@app.route("/get_macros", methods=["POST"])
def get_macros_from_name():
    if not gemini_model:
        return jsonify({"error": "Gemini modeli başlatılamadı."}), 503
    data = request.get_json()
    if not data or 'food_name' not in data:
        return jsonify({"error": "Lütfen 'food_name' anahtarıyla bir yemek adı gönderin."}), 400
    food_name = data['food_name']

    prompt_template = """Kullanıcının verdiği yemek adının 100 gramı için ortalama kalori (kcal), karbonhidrat (carbs), protein ve yağ (fat) değerlerini bul. Cevabını SADECE ve SADECE aşağıdaki JSON formatında ver. Başka hiçbir açıklama, selamlama veya metin ekleme. Eğer yemeği bulamazsan veya emin değilsen, tüm değerleri 0 olarak döndür.

Örnek Çıktı:
{{
  "calories": 250,
  "carbs": 30.5,
  "protein": 15.2,
  "fat": 8.1
}}

Yemek Adı: "{food_name}"
"""

    final_prompt = prompt_template.format(food_name=food_name)
    try:
        response = gemini_model.generate_content(final_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        macros = json.loads(cleaned_response)
        return jsonify(macros)
    except json.JSONDecodeError:
        return jsonify({"error": "Makro bilgisi alınırken format hatası oluştu."}), 500
    except Exception as e:
        app.logger.error(f"Gemini Makro Bulucu hatası: {e}", exc_info=True)
        return jsonify({"error": f"Makro bilgisi alınırken bir hata oluştu: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)