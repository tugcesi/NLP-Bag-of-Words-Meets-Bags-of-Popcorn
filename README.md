# 🎬 NLP — Bag of Words Meets Bags of Popcorn

IMDB film yorumlarını **pozitif** veya **negatif** olarak sınıflandıran bir duygu analizi uygulaması.  
Model, **Bag of Words** yöntemi ve **Keras** kullanılarak eğitilmiştir. Arayüz **Streamlit** ile oluşturulmuştur.

---

## 📁 Proje Yapısı

```
├── app.py                              # Streamlit uygulaması
├── bagofwords_model.h5                 # Eğitilmiş Keras modeli
├── vectorizer.pkl                      # Eğitilmiş CountVectorizer
├── requirements.txt                    # Bağımlılıklar
├── save_vectorizer.py                  # Vectorizer'ı kaydetmek için yardımcı script
└── bag-of-words-meets-bags-of-popcorn.ipynb  # Model eğitim notebook'u
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Repoyu klonlayın
```bash
git clone https://github.com/tugcesi/NLP-Bag-of-Words-Meets-Bags-of-Popcorn.git
cd NLP-Bag-of-Words-Meets-Bags-of-Popcorn
```

### 2. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı başlatın
```bash
streamlit run app.py
```

---

## 🧠 Model Hakkında

| Özellik | Detay |
|---|---|
| Yöntem | Bag of Words (CountVectorizer) |
| Model | Keras Dense Neural Network |
| Veri Seti | IMDB 25.000 film yorumu |
| Görev | Binary Sentiment Classification |
| Çıktı | Pozitif 😊 / Negatif 😞 |

---

## 🛠️ Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- NLTK
- BeautifulSoup4

---

## ⚠️ Önemli Not

`vectorizer.pkl` dosyasını oluşturmak için `save_vectorizer.py` scriptini notebook ortamınızda çalıştırın ve üretilen dosyayı proje dizinine ekleyin.
