from transformers import pipeline

# Анализирует тональность текста с использованием предварительно обученной модели.
# Параметры:
# - text (str): Входной текст для анализа.
# Возвращает:
# - dict: Словарь с меткой и вероятностью тональности.

def analyze_sentiment(text):
    classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
    result = classifier(text)
    return {"label": result[0]["label"], "score": result[0]["score"]}

text1 = "Я обожаю инженерию машинного обучения!"
text2 = "Я ненавижу инженерию машинного обучения!"

result1 = analyze_sentiment(text1)
result2 = analyze_sentiment(text2)

print(f"Тональность текста 1: {result1['label']}, вероятность: {result1['score']:.4f}")
print(f"Тональность текста 2: {result2['label']}, вероятность: {result2['score']:.4f}")
