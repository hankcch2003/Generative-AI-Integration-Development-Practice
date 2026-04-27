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

# 初始化英文 Tokenizer 並進行詞彙擬合
eng_tokenizer = Tokenizer(filters = '')
eng_tokenizer.fit_on_texts([p[0] for p in pairs])

# 定義字元切割函式，用於處理中文
def char_tokenize(text):
    return list(text)

# 將中文句子拆解為字元並用空格隔開，以便 Tokenizer 處理
chi_texts = [" ".join(char_tokenize(p[1])) for p in pairs]
chi_tokenizer = Tokenizer(filters = '')
chi_tokenizer.fit_on_texts(chi_texts)

# 序列化：將文字轉換成數字序列
X = eng_tokenizer.texts_to_sequences([p[0] for p in pairs])
y = chi_tokenizer.texts_to_sequences(chi_texts)

# 計算最大長度：找出英文與中文序列的最長值作為 Padding 基準
max_len_eng = max(len(seq) for seq in X)
max_len_chi = max(len(seq) for seq in y)

# 規劃相同長度：使用 post padding 將所有序列補齊至相同長度
X = pad_sequences(X, maxlen = max_len_eng, padding = 'post') 
y = pad_sequences(y, maxlen = max_len_chi, padding = 'post')

# 規劃訓練用的模型
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, AdditiveAttention
from tensorflow.keras.models import Model

# 設定隱藏層維度，影響模型的特徵記憶能力
latent_dim = 64

# 定義 Encoder：將輸入的英文序列編碼為特徵向量與狀態
encoder_inputs = Input(shape = (max_len_eng,))
x = tf.keras.layers.Embedding(len(eng_tokenizer.word_index) + 1, 64)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_sequences = True, return_state = True)(x)

# 定義 Decoder：接收 Encoder 的狀態並準備進行解碼
decoder_inputs = Input(shape = (max_len_chi,))
y_emb = tf.keras.layers.Embedding(len(chi_tokenizer.word_index) + 1, 64)(decoder_inputs)
decoder_outputs = LSTM(latent_dim, return_sequences = True)(y_emb, initial_state = [state_h, state_c])

# 加入注意力機制 (Attention)：讓模型學習輸出時應該關注輸入的哪些部分
attention_layer = AdditiveAttention()
context, attention_scores = attention_layer(
    [decoder_outputs, encoder_outputs],
    return_attention_scores = True
)

# 最終輸出層：使用 Softmax 計算每個中文字元的機率分布
outputs = Dense(len(chi_tokenizer.word_index) + 1, activation = 'softmax')(context)
model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.summary()

from tensorflow.keras.utils import plot_model

# 準備訓練資料並執行模型擬合 (Training)
y_target = np.expand_dims(y, -1)
history = model.fit(
    [X, y],
    y_target,
    epochs = 300,
    verbose = 1
)

# 建立預估模型：包含輸出結果與注意力分數，用於後續視覺化分析
attention_model = Model(
    inputs = model.inputs,
    outputs = [model.output, attention_scores]
)

# 測試階段：設定測試輸入並轉換為模型可接收的格式
test_input = "i live in taipei and love night markets"
test_seq = eng_tokenizer.texts_to_sequences([test_input])
test_seq = pad_sequences(test_seq, maxlen = max_len_eng, padding = 'post')
decoder_input = np.zeros((1, max_len_chi))
pred, attn = attention_model.predict([test_seq, decoder_input])

# 準備熱圖視覺化所需資料
import matplotlib.pyplot as plt
import seaborn as sns
input_words = test_input.split()

# 建立數字與中文字元的反向對照表
reverse_chi_word_index = dict((i, char) for char, i in chi_tokenizer.word_index.items())

# 取得預測結果中機率最大的字元索引值
predicted_indices = np.argmax(pred[0], axis = -1)

# Y 軸：使用預測出的標籤將數字索引轉回中文
output_words = [reverse_chi_word_index.get(idx, "") for idx in predicted_indices]

# X 軸：將英文單字補齊空白以對應熱圖矩陣寬度
display_input_words = input_words + [""] * (max_len_eng - len(input_words))

from matplotlib import font_manager
font_path = r'E:\Python\Python4-人工智慧整合開發實務\20260402\NotoSerifCJKtc-Black.otf'
font_prop = font_manager.FontProperties(fname = font_path) # 產生一個字型物件

# 繪製注意力熱圖 (Attention Map)
plt.figure(figsize = (10, 8))
sns.heatmap(attn[0],
            xticklabels = display_input_words,
            yticklabels = output_words,
            annot = True,
            cmap = "Blues")

plt.xlabel("Input (English)")
plt.ylabel("Output (Chinese)")
plt.title("顯示每一個字被關注的程度", fontproperties = font_prop)
plt.yticks(fontproperties = font_prop, rotation = 0) # 設定左側中文字型
plt.show()

# 輸出詳細預測機率與注意力權重數據
print('pred =', pred)
print('attn =', attn)
predicted_chars = []

# 過濾 Padding (索引 0) 並將結果組合成最終翻譯字串
for idx in predicted_indices:
    if idx == 0: 
        continue
    char = reverse_chi_word_index.get(idx, "")
    predicted_chars.append(char)

# 格式化輸出：每個字中間留空格
translation = " ".join(predicted_chars)

print("英文輸入:", test_input)
print("模型翻譯:", translation)