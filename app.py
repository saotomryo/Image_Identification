import torch
from torchvision import transforms
from torchvision.models import mobilenetv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

import streamlit as st
from PIL import Image

# GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([ # 検証データ用の画像の前処理
    transforms.RandomResizedCrop(size=(224,224),scale=(1.0,1.0),ratio=(1.0,1.0)), # アスペクト比を保って画像をリサイズ
    #transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# モデルクラスの宣言

mob_model = mobilenetv2.mobilenet_v2(pretrained=False)

class Mobilenetv2(nn.Module):
    def __init__(self, mob_model):
        super(Mobilenetv2, self).__init__()
        self.vit = mob_model
        self.fc = nn.Linear(1000, 10)

    def forward(self, input_ids):
        states = self.vit(input_ids)
        states = self.fc(states)
        return states

st.title('画像判定アプリ')

st.markdown('学習の実施は[こちら](https://github.com/saotomryo/Image_Identification/blob/master/Use_MobelenetV2.ipynb)')
upload_model = st.file_uploader('学習したAIモデルをアップロードしてください(アップロードしない場合は、事前学習された内容で判定します。)',type=['pth'])


if upload_model is not None:
    #net.load_state_dict(torch.load(upload_model,map_location=torch.device('cpu')))
    net = torch.load(upload_model)
    features = net.categories
else:
    net = mobilenetv2.mobilenet_v2(pretrained=True)
    json_open = open('imagenet1000_clsidx_to_labels.json', 'r')
    json_load = json.load(json_open)
    #print(json_load)
    #features = json_open[i]
    #features = [i for i in range(1000)]


uploaded_file = st.file_uploader('判定する写真をアップロードが撮影してください。', type=['jpg','png','jpeg'])
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    data = torch.reshape(transform(img),(-1,3,224,224))

    net.eval()

    with torch.no_grad():
        out = net(data)
        predict = out.argmax(dim=1)
        #st.write(out)

    st.markdown('認識結果')

    if upload_model is not None:
        st.write(features[predict.detach().numpy()[0]])
    else:
        i = predict.detach().numpy()[0]

        st.write(json_load[str(i)])

    st.image(img)
