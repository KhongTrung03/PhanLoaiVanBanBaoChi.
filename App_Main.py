from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
import re
from transformers import AutoTokenizer

app = Flask(__name__)

# 🔹 Load PhoBERT Tokenizer
tokenizer = AutoTokenizer.from_pretrained(r"phobert_tokenizer")

# 🔹 Load mô hình đã train
model_path = r"Bert_main"
model_classifier = tf.saved_model.load(model_path)
infer = model_classifier.signatures["serving_default"]

# 🔹 Load LabelEncoder
label_encoder = joblib.load(r"label_encoder.pkl")

# 🔹 Định nghĩa max_len
MAX_LEN = 256  

def split_text_into_chunks(text, tokenizer, max_len=MAX_LEN):
    """Chia văn bản thành các đoạn nhỏ hợp lý (không cắt ngang câu)"""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Giữ dấu chấm ở cuối câu
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(tokenized_sentence)
        
        if current_length + sentence_length > max_len:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = tokenized_sentence
            current_length = sentence_length
        else:
            current_chunk.extend(tokenized_sentence)
            current_length += sentence_length
            
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def encode_texts(tokenized_texts, tokenizer, max_length=MAX_LEN):
    """Mã hóa tokenized texts thành input_ids và attention_mask"""
    tokenized_texts = [chunk[:max_length] for chunk in tokenized_texts]  # Cắt tối đa 256 tokens
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_texts, maxlen=max_length, dtype="int32", padding="post", truncating="post", value=pad_token_id
    )
    
    attention_masks = np.where(input_ids != pad_token_id, 1, 0)
    
    return input_ids, attention_masks

def predict_text_chunks(input_ids_chunks, attention_mask_chunks):
    all_predictions = []
    
    for i in range(len(input_ids_chunks)):
        inputs = {
            "input_ids": tf.convert_to_tensor([input_ids_chunks[i]], dtype=tf.int32),
            "attention_mask": tf.convert_to_tensor([attention_mask_chunks[i]], dtype=tf.int32),
        }
        
        # Dự đoán với mô hình PhoBERT
        outputs = infer(**inputs)
        output_key = list(outputs.keys())[0]
        predictions = outputs[output_key].numpy()
        
        predicted_label_id = np.argmax(predictions, axis=1)[0]
        all_predictions.append(predicted_label_id)
    
    # Chọn nhãn xuất hiện nhiều nhất
    final_label_id = np.bincount(all_predictions).argmax()
    return label_encoder.inverse_transform([final_label_id])[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.form['content'].strip()
        if content == "":
            return render_template('index.html', error="Trường này không được để trống")

        if model_classifier is None:
            return render_template('index.html', error="Lỗi khi tải mô hình, vui lòng kiểm tra lại!")
        
        # 🔹 Chia văn bản thành các đoạn nhỏ
        token_chunks = split_text_into_chunks(content, tokenizer, max_len=MAX_LEN)
        
        # 🔹 Mã hóa các đoạn văn thành input_ids và attention_mask
        input_ids_chunks, attention_mask_chunks = encode_texts(token_chunks, tokenizer, max_length=MAX_LEN)

        # 🔹 Dự đoán từng đoạn và lấy nhãn phổ biến nhất
        predicted_label = predict_text_chunks(input_ids_chunks, attention_mask_chunks)
        
        return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
