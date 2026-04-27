pairs = [
    ("i live in taipei", "我住在台北"),
    ("i love night markets", "我喜歡夜市"),
    ("taipei is beautiful", "台北很美"),
    ("i go to night markets", "我去夜市"),
    ("night markets", "夜市"),    
    ("i live in taipei and love night markets", "我住在台北而且喜歡夜市"),
    ("Yilan is a beautiful county with many delicious foods", "宜蘭是一個很美的縣市且有很多美食")
]

# 文字分詞與向量序列化處理
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 初始化英文分詞器並對 pairs 中的英文句子進行擬合
eng_tokenizer = Tokenizer(filters = '')
eng_tokenizer.fit_on_texts([p[0] for p in pairs])

def char_tokenize(text):
    return list(text)

# 將中文句子進行字元切割並重新組合，以便後續 Tokenizer 處理
chi_texts = [" ".join(char_tokenize(p[1])) for p in pairs]
chi_tokenizer = Tokenizer(filters = '')
chi_tokenizer.fit_on_texts(chi_texts)

# 序列化：將文字轉換為對應的數字索引序列
X = eng_tokenizer.texts_to_sequences([p[0] for p in pairs])
y = chi_tokenizer.texts_to_sequences(chi_texts)

# 計算最大長度：找出英文與中文序列的最長值作為統一長度的基準
max_len_eng = max(len(seq) for seq in X)
max_len_chi = max(len(seq) for seq in y)

# 規劃相同長度：使用 post padding 將所有序列補齊至相同長度
X = pad_sequences(X, maxlen = max_len_eng, padding = 'post')
y = pad_sequences(y, maxlen = max_len_chi, padding = 'post')

# 規劃訓練用的模型
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, AdditiveAttention
from tensorflow.keras.models import Model

# 設定隱藏狀態的維度大小
latent_dim = 64

# 定義 Encoder (編碼器)：將英文輸入序列壓縮成具備上下文資訊的特徵向量
encoder_inputs = Input(shape = (max_len_eng,))
x = tf.keras.layers.Embedding(len(eng_tokenizer.word_index) + 1, 64)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_sequences = True, return_state = True)(x)

# 定義 Decoder (解碼器)：接收編碼器的狀態並結合注意力機制產出中文預測
decoder_inputs = Input(shape = (max_len_chi,))
y_emb = tf.keras.layers.Embedding(len(chi_tokenizer.word_index) + 1, 64)(decoder_inputs)
decoder_outputs = LSTM(latent_dim, return_sequences = True)(y_emb, initial_state = [state_h, state_c])

# 加入注意力機制層，計算解碼過程與編碼輸出的關聯性分數
attention_layer = AdditiveAttention()
context, attention_scores = attention_layer(
    [decoder_outputs, encoder_outputs],
    return_attention_scores = True
)

# 使用 Dense 層與 Softmax 函數計算最終中文字元的機率分布
outputs = Dense(len(chi_tokenizer.word_index) + 1, activation = 'softmax')(context)
model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.summary()

from tensorflow.keras.utils import plot_model

# 繪製模型架構圖並儲存成圖檔
plot_model(model, to_file = 'model1.png', show_shapes = True, show_layer_names = True)

# 準備訓練資料並執行模型訓練
y_target = np.expand_dims(y, -1)
history = model.fit(
    [X, y],
    y_target,
    epochs = 300,
    verbose = 1
)

# 建立推論用的注意力模型，以便提取翻譯結果與注意力權重矩陣
attention_model = Model(
    inputs = model.inputs,
    outputs = [model.output, attention_scores]
)

# 繪製推論模型的架構圖
plot_model(attention_model, to_file = 'attention_model.png', show_shapes = True, show_layer_names = True)

# 測試階段：設定測試輸入並執行預測
test_input = "night markets"
test_seq = eng_tokenizer.texts_to_sequences([test_input])
test_seq = pad_sequences(test_seq, maxlen = max_len_eng, padding = 'post')
decoder_input = np.zeros((1, max_len_chi))
pred, attn = attention_model.predict([test_seq, decoder_input])

# 處理預測結果與標籤
reverse_chi_word_index = dict((i, char) for char, i in chi_tokenizer.word_index.items())
predicted_indices = np.argmax(pred[0], axis = -1)
predicted_chars = [reverse_chi_word_index.get(idx, "") for idx in predicted_indices if idx != 0]

# 設定熱圖座標
input_words = test_input.split()
output_words = predicted_chars # 使用實際預測出的字，圖才不會空白

import matplotlib.pyplot as plt
import seaborn as sns
input_words = test_input.split()

# 建立索引與文字的對應表以便反解翻譯文字
output_words = list("我住在台北而且喜歡夜市")

from matplotlib import font_manager
font_path = r'E:\Python\Python4-人工智慧整合開發實務\20260402\NotoSerifCJKtc-Black.otf'
font_prop = font_manager.FontProperties(fname = font_path) # 產生一個字型物件

# 繪製注意力權重熱圖 (Attention Map)
plt.figure(figsize = (10, 6))
sns.heatmap(attn[0][:len(output_words), :len(input_words)],
            xticklabels = input_words,
            yticklabels = output_words,
            cmap = "Blues")

plt.xlabel("Input (English)")
plt.ylabel("Output (Chinese)")
plt.title("顯示每一個字被關注的程度", fontproperties = font_prop)
plt.yticks(fontproperties = font_prop, rotation = 0) # 設定左側中文字型
plt.show()

# 輸出預測結果與注意力權重矩陣數據
print('pred =', pred)
print('attn =', attn)

reverse_chi_word_index = dict((i, char) for char, i in chi_tokenizer.word_index.items())
predicted_indices = np.argmax(pred[0], axis = -1)
predicted_chars = []

# 將預測結果轉回中文文字並過濾掉補齊用的 0 (Padding)
for idx in predicted_indices:
    if idx == 0: 
        continue
    char = reverse_chi_word_index.get(idx, "")
    predicted_chars.append(char)

# 格式化輸出：每個字中間留空格
translation = " ".join(predicted_chars)

print("英文輸入:", test_input)
print("模型翻譯:", translation)