# Image_Identification
Image Identification

# 利用方法

1.画像分類するデータを以下の形式で用意して、ZIPファイルに圧縮します。

　トップのフォルダ  
 　- 分類したラベル名のフォルダ1  
     - 画像ファイル  
 　- 分類したラベル名のフォルダ2  
     - 画像ファイル  
  .  
  .  
  .  
  
2.「Use_MobelenetV2.ipynb」をgoogle colaboratoryで開き、1で用意したZIPファイルを使って学習を実施ます。
　　（「全てのセルの実行」を選択し、1.で作成したZIPファイルをアップロードするだけで学習済みのモデルがダウンロードされます。）

3.requirement.txtに記載のモジュールをインストールします。

4.Pythonの実行環境上のコマンドプロンプトまたはターミナルで、「streamlit run app.py」と入力してエンターします。

5.Webサービスが開始され、デフォルトで「http://localhost:8501  」  でアプリケーションにアクセス出来ます。

6.2で作成し、ダウンロードした学習モデルをアップロードします。

7.判別する画像をアップロードまたは、撮影します。

アプリケーションは、以下で公開しています。

https://share.streamlit.io/saotomryo/image_identification/app.py




