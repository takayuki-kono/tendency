# Google連携セットアップ手順

本プロジェクトで画像管理をGoogle Spreadsheetと連携させるためのセットアップ手順です。

## 1. Google Cloud Platform (GCP) の設定

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセスします。
2. 左上のプロジェクト選択プルダウンから「新しいプロジェクト」を作成します（名前は `tendency-project` など任意）。
3. **APIの有効化**:
   - 左メニューの「APIとサービス」>「ライブラリ」へ移動。
   - 以下の2つを検索して「有効にする」をクリックします。
     - **Google Drive API**
     - **Google Sheets API**

## 2. サービスアカウントの作成とキー取得

1. 「APIとサービス」>「認証情報」へ移動します。
2. 「認証情報を作成」>「サービスアカウント」を選択。
3. 名前を入力（例: `sheet-sync`）して「作成して続行」。権限は「オーナー」または「編集者」を選択して完了。
4. 作成されたサービスアカウントのメールアドレス（例: `sheet-sync@tendency-project...`）をコピーして控えておきます。
5. 作成したサービスアカウントをクリックし、「キー」タブへ移動。
6. 「鍵を追加」>「新しい鍵を作成」>「JSON」を選択して作成。
7. 自動的にJSONファイルがダウンロードされます。

## 3. ファイルの配置

1. ダウンロードしたJSONファイルを `service_account.json` にリネームします。
2. 本プロジェクトの `config` フォルダに配置します。
   - 配置場所: `d:\tendency\config\service_account.json`

## 4. スプレッドシートの準備

1. [Google Sheets](https://docs.google.com/spreadsheets/) で新しいシートを作成します。
2. 右上の「共有」ボタンをクリック。
3. 手順2-4で控えた**サービスアカウントのメールアドレス**を入力し、「編集者」として送信（招待）します。
4. ブラウザのURLバーから `Spreadsheet ID` をコピーします。
   - `https://docs.google.com/spreadsheets/d/` **ここがIDです** `/edit...`

## 5. 実行前の準備

以下のコマンドでライブラリをインストールします。

```cmd
pip install gspread google-api-python-client oauth2client
```

## 6. スクリプトの実行

スクリプト内の `SPREADSHEET_ID` 変数に手順4で取得したIDを設定して実行します。

```cmd
python util/sync_sheets_images.py
```
