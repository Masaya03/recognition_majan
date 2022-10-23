import streamlit as st
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys
import os
import shutil
import math
from PIL import Image
from model import recog_majan

st.set_page_config(page_title="メインページ")

page_name = ["main", "sample"]
page = st.sidebar.selectbox("page",page_name)
labels = []
with open("./label/labels2.txt",'r',encoding="utf-8") as f:
    for line in f:
        labels.append(line)
# (labels[0])

#mainページの表示
if page == "main":
    st.title("麻雀牌の画像認識アプリ")

    st.markdown("#### 試作段階であり、一つの牌しか認識できません")
    st.markdown("牌のサンプル画像はsidebar pageのsampleにあります")

    uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type="jpg")
    st.write("")
    st.write("")

    if uploaded_file != None:
        image = Image.open(uploaded_file)

        #モデルの読み込みと予測結果の表示
        model = recog_majan("my_model.h5", "model")
        pred = model.predict(image)
        st.write("予測結果は"+pred+"です")
        
        st.image(
            image,
            caption=f"You amazing image has shape",
            use_column_width=True
        )

#sample画面
elif page == "sample":
    st.title("サンプル画像")
    sample_dirname = "./sample_image"
    images_name = glob.glob(sample_dirname+"/*")

    if len(images_name) > 0:
        image_num = st.selectbox("画像番号",
            range(len(images_name))
            # labels
        )

    st.write("")
    st.write("")

    if image_num != None:
        # images_name[image_num]
        image = Image.open(images_name[image_num])

        #モデルの読み込みと表示
        model = recog_majan("my_model.h5", "model")
        pred = model.predict(image)
        st.write("予測結果は"+pred+"です")

        st.image(
            image,
            caption=f"You amazing image has shape",
            use_column_width=True,
            )