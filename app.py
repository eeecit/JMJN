import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# 1. 구글 드라이브에서 모델을 자동으로 다운로드하고 캐시에 로드
@st.cache_resource
def load_my_model():
    # 👇 여기에 본인의 구글 드라이브 파일 ID를 넣으세요!
    file_id = '112l8nLCbaBYQpbyCWehZbiXJeAb6yHfa'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'best_model.keras'
    
    # 파일이 아직 없다면 드라이브에서 다운로드
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        
    return tf.keras.models.load_model(output)

model = load_my_model()

# 2. 웹사이트 화면 구성
st.title("📸 이제노 vs 나재민 판별기")
st.write("사진을 업로드하면 어떤 멤버인지 AI가 판별해 줍니다!")

uploaded_file = st.file_uploader("이미지를 선택하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    
    with st.spinner('AI가 분석 중입니다...'):
        image_resized = image.resize((300, 300))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)[0][0]
        
        st.subheader("🤖 예측 결과")
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.success(f"**이제노**일 확률이 **{confidence:.2f}%** 입니다!")
        else:
            confidence = prediction * 100
            st.info(f"**나재민**일 확률이 **{confidence:.2f}%** 입니다!")