import pickle

# Notebook'unuzda CountVectorizer'ı eğittikten sonra bu kodu çalıştırın.
# "vectorizer" değişken adını kendi notebook'unuzdaki isimle değiştirin.

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ vectorizer.pkl başarıyla kaydedildi!")
