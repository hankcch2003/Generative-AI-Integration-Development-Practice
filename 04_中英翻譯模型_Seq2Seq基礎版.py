pairs = [
    ("i live in taipei", "我住在台北"),
    ("i love night markets", "我喜歡夜市"),
    ("taipei is beautiful", "台北很美"),
    ("i go to night markets", "我去夜市"),
    ("i live in taipei and love night markets", "我住在台北而且喜歡夜市")
]

# 文字分詞與向量序列化處理
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

print('以下為字串序列化處理過程，計算出最大長度後規劃相同長度')

# 初始化英文分詞器並學習詞彙
eng_tokenizer = Tokenizer(filters = '')
eng_tokenizer.fit_on_texts([p[0] for p in pairs])

def char_tokenize(text):
    return list(text)

# 處理中文：將句子拆解為字元並用空格隔開，以便 Tokenizer 進行字元級分詞
chi_texts = [" ".join(char_tokenize(p[1])) for p in pairs]
chi_tokenizer = Tokenizer(filters = '')
chi_tokenizer.fit_on_texts(chi_texts)

# 序列化：將文字轉換為模型可處理的數字編號序列
X = eng_tokenizer.texts_to_sequences([p[0] for p in pairs])
y = chi_tokenizer.texts_to_sequences(chi_texts)

# 計算最大長度：取得英中序列的最長長度作為填充標準
max_len_eng = max(len(seq) for seq in X)
max_len_chi = max(len(seq) for seq in y)

# 規劃相同長度：使用 post padding 補齊序列，確保輸入維度一致
X = pad_sequences(X, maxlen = max_len_eng, padding = 'post')
y = pad_sequences(y, maxlen = max_len_chi, padding = 'post')

# 規劃訓練用的模型
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# 增加維度以避免過度壓縮與資訊孤島問題，提升特徵表達能力
latent_dim = 64

print('規劃 encoder 編碼處理，也就是英文資料')

# 定義 Encoder：負責將英文輸入轉換為上下文特徵與狀態向量 (state_h, state_c)
encoder_inputs = Input(shape = (max_len_eng,))
x = tf.keras.layers.Embedding(len(eng_tokenizer.word_index) + 1, 64)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_sequences = True, return_state = True)(x)

print('規劃 decoder 解碼處理，也就是中文')

# 定義 Decoder：接收編碼器的狀態，並逐步產生對應的中文字
decoder_inputs = Input(shape = (max_len_chi,))
y_emb = tf.keras.layers.Embedding(len(chi_tokenizer.word_index) + 1, 64)(decoder_inputs)
decoder_outputs = LSTM(latent_dim, return_sequences = True)(y_emb, initial_state = [state_h, state_c])

print('假設不加入 Attention')

# 此處不使用 Attention 運算，直接將 decoder_outputs 送入全連接層預測字元機率分布
outputs = Dense(len(chi_tokenizer.word_index) + 1, activation = 'softmax')(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.summary()

from tensorflow.keras.utils import plot_model

# 產生主模型架構圖
plot_model(model, to_file = 'model2.png', show_shapes = True, show_layer_names = True)

# 訓練模型：執行 300 次訓練 (Epochs)，使模型能充分學習並記憶小樣本數據的對應關係
y_target = np.expand_dims(y, -1)
history = model.fit(
    [X, y],
    y_target,
    epochs = 300, 
    verbose = 1
)

print('規劃評估模型，評估不等於訓練')

# 獨立的推論用 Encoder 模型
encoder_model = Model(encoder_inputs, [state_h, state_c])

# 獨立的推論用 Decoder 模型結構
decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 從訓練好的主模型中擷取層來重用權重
dec_emb2 = model.layers[3](decoder_inputs)
decoder_outputs2, state_h2, state_c2 = model.layers[4](
    dec_emb2, initial_state = decoder_states_inputs
)
decoder_outputs2 = model.layers[-1](decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2, state_h2, state_c2]
)

# 產生推論模型的結構圖
plot_model(encoder_model, to_file = 'encoder_model.png', show_shapes = True, show_layer_names = True)
plot_model(decoder_model, to_file = 'decoder_model.png', show_shapes = True, show_layer_names = True)

print('規劃預估函數')
def decode_sequence(input_text):
    # 將英文輸入文本轉換為序列格式
    seq = eng_tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen = max_len_eng, padding = 'post')
    
    # 執行預測並隱藏進度條，使輸出結果更簡潔
    states = encoder_model.predict(seq, verbose = 0)
    target_seq = np.zeros((1, 1)) # 設定初始輸入標記 (通常為 0)
    
    stop = False
    predicted_chars = []
    
    while not stop:
        # 逐步進行解碼預測
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states, verbose = 0
        )
        
        # 取得機率最大的下一個字元索引
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # 若預測到索引 0 (Padding)，代表模型判斷句子已結束，提前跳出迴圈
        if sampled_token_index == 0:
            stop = True
            break
            
        # 將索引還原為中文字元
        sampled_word = chi_tokenizer.index_word.get(sampled_token_index, "")
        
        if sampled_word != "":
            predicted_chars.append(sampled_word)
        
        # 更新輸入與狀態，為下一個字元的預測做準備
        target_seq = np.array([[sampled_token_index]])
        states = [h, c]
        
        # 達到設定的最大中文長度則停止
        if len(predicted_chars) >= max_len_chi:
            stop = True

    # 格式化輸出：每個字中間留空格
    return " ".join(predicted_chars)

# 測試預估結果
test_input = "i live in taipei and love night markets"
result = decode_sequence(test_input)

print("英文輸入:", test_input)
print("模型翻譯:", result)