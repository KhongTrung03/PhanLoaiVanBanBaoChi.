import pandas as pd
from pyvi import ViTokenizer
import re

class TextPreprocessor:
    def __init__(self, stopwords_file):
        self.stopword_set = self.load_stopwords(stopwords_file)

    def load_stopwords(self, stopwords_file):
        stopword_df = pd.read_csv(stopwords_file, header=None, names=['stopword'])
        return set(stopword_df['stopword'])

    def remove_stopwords(self, line):
        words = [word for word in line.strip().split() if word not in self.stopword_set]
        return ' '.join(words)

    # Danh sách nhãn cần giữ lại đại từ nhân xưng
    keep_pronouns_labels = {'Góc nhìn'}

    # Danh sách đại từ nhân xưng tiếng Việt (loại bỏ `|` dư thừa)
    pronouns = r'\b(tôi|tao|mình|chúng tôi|bọn tao|bọn tôi|bọn tớ|bọn mình|ta|tớ|bạn|cậu|mày|mi|bác|thầy|cô|chú|dì|cậu|mợ|ông|bà|anh|chị|chúng mày|các bạn|các ông|các bà|em|hắn|hắn ta|hắn ấy|nó|nó ta|nó ấy|họ|y|lão|cụ|thím|con|cháu)\b'

    def preprocess_text(self, text, label):
        text = text.lower()  # Chuyển thành chữ thường
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' ', text)  # Xóa email
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)  # Xóa URL
        text = re.sub(r'\b\d{1,2}[-/]\d{1,2}([-/]\d{2,4})?\b', ' ', text)  # Xóa ngày tháng
        text = re.sub(r'(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)', ' ', text)  # Tách chữ dính số
        text = re.sub(r'\d+', '', text).strip()  # Xóa số
        text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)  # Xóa dấu câu
        text = re.sub(r'\s+\b[a-zA-Z]\b\s+', ' ', text)  # Xóa ký tự đơn lẻ
        text = re.sub(r'\s+', ' ', text).strip()  # Xóa khoảng trắng dư thừa

        # Chỉ loại bỏ đại từ nhân xưng nếu nhãn không thuộc nhóm cần giữ
        if label not in self.keep_pronouns_labels:
            text = re.sub(self.pronouns, '', text, flags=re.IGNORECASE).strip()
        text = ViTokenizer.tokenize(text)
        text = self.remove_stopwords(text)
        return text

    def preprocess_data(self, data):
        data['Content'] = data.apply(lambda row: self.preprocess_text(row['Content'], row['Label']), axis=1)
        return data


