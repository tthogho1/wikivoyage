以下は、**英語Wikivoyageの現行版ダンプから記事内の全画像を取得し、SigLIP2で埋め込み、Zilliz CloudへページURL・タイトル付きで登録するシステム**の要件・仕様です。プログラムではなく、実装・発注・レビューに使える仕様書として整理します。

## 1. システム目的

英語Wikivoyageの記事に掲載された画像を収集し、画像内容をベクトル化してZilliz Cloudに登録する。これにより、画像類似検索、テキスト検索との組み合わせ、観光地・目的地・ランドマークのビジュアル検索などを可能にする。

対象は **英語版Wikivoyage（`enwikivoyage`）の現行版ダンプ**、対象画像は **各記事に含まれる全画像**、埋め込みモデルは **`google/siglip2-so400m-patch16-384`** とする。

Wikimediaのダンプには記事のwikitextとメタデータは含まれるが、画像ファイル本体は含まれない。そのため、ダンプから画像ファイル名を抽出した後、Wikimedia Commons APIで画像URL・ライセンス等を取得し、画像をダウンロードする必要がある。 [dumps.wikimedia](https://dumps.wikimedia.org/enwikivoyage/20261101/)

## 2. 対象範囲

### 2.1 対象データ

| 項目 | 仕様 |
|---|---|
| Wiki | 英語版Wikivoyage（`enwikivoyage`） |
| ダンプ種別 | 現行版の `pages-articles` ダンプ |
| ダンプ形式 | MediaWiki XML、bzip2圧縮 |
| ダンプ日付 | 実行時に指定可能。例：`20261101` |
| 対象ページ | 名前空間 `0` の通常記事 |
| 除外ページ | リダイレクト、利用者ページ、Template、Category、Talk、Fileページ |
| 対象画像 | 各記事のwikitextに埋め込まれた全画像 |
| 画像ソース | Wikimedia Commons |
| 埋め込み対象 | 画像ファイル |
| ベクトルDB | Zilliz Cloud |

WikivoyageのダンプはWikimediaのダンプサービスから入手でき、英語版Wikivoyageのダンプは `https://dumps.wikimedia.org/enwikivoyage/<YYYYMMDD>/` 配下に提供される。 [dumps.wikimedia](https://dumps.wikimedia.org/enwikivoyage/20261101/)

### 2.2 画像の抽出対象

以下の形式を抽出対象とする。

- `[[File:Example.jpg]]`
- `[[Image:Example.jpg]]`
- `[[File:Example.jpg|thumb|Caption text]]`
- `[[File:Example.jpg|left|200px|Caption text]]`
- Infoboxやテンプレート内で指定される画像
- Gallery形式の画像
- 記事本文・Infobox・ギャラリーに現れるすべての画像参照

抽出時に、大文字・小文字、`File:` と `Image:` の違い、日本語名前空間の有無を吸収する必要がある。ただし英語版が対象のため、基本は `File:` と `Image:` を扱う。

## 3. 機能要件

### 3.1 ダンプ取得機能

システムは、指定されたダンプ日付の英語Wikivoyage現行版ダンプをダウンロードできること。

- ダンプURLは `https://dumps.wikimedia.org/enwikivoyage/<YYYYMMDD>/enwikivoyage-<YYYYMMDD>-pages-articles.xml.bz2` を基準とする。
- ダウンロード済みファイルがある場合は再ダウンロードしない。
- ダウンロード中断時は再開または再実行できること。
- ダウンロード後、Wikimediaが提供するMD5またはSHA1チェックサムで整合性を検証できること。
- ダウンロード時には、プロジェクト名・連絡先を含む識別可能な `User-Agent` を送信すること。
- 同時接続数は制限し、Wikimediaのレート制限に従うこと。

Wikimediaはダンプダウンロード時のUser-Agentポリシーとレート制限を明示しており、同時接続数の上限や過度な並列取得を避けるよう求めている。

### 3.2 記事抽出機能

システムは、ダウンロードしたXMLダンプをストリーム解析し、記事単位で以下を抽出できること。

- `page_id`
- `page_title`
- `namespace`
- リダイレクトフラグ
- 記事本文のwikitext
- 最終更新日時
- ダンプ内のバージョン情報

抽出条件は次のとおり。

- `namespace = 0` のみ処理する。
- リダイレクトページは除外する。
- 空ページ、削除済みページ、画像参照を持たないページはスキップする。
- 記事ごとに、画像抽出結果を一時保存または逐次処理できること。
- ダンプ全体を一度にメモリへ読み込まず、ストリーム処理すること。

### 3.3 画像抽出機能

システムは、各記事のwikitextから、記事内で参照される全画像ファイル名を抽出できること。

抽出項目は以下とする。

| 項目 | 内容 |
|---|---|
| `file_title` | `File:Example.jpg` 形式の正規化済みファイル名 |
| `caption` | 画像リンクに付随するキャプション |
| `position` | 記事内での出現順 |
| `context` | 画像周辺のwikitext断片 |
| `source_page_id` | 画像が参照された記事のページID |
| `source_page_title` | 画像が参照された記事タイトル |

仕様は以下とする。

- 同一記事内で同じ画像が複数回使われる場合、`file_title` 単位で重複排除する。
- キャプションは可能な範囲で抽出する。
- キャプションが取得できない場合も、画像登録は継続する。
- 1記事あたりの画像数に上限を設けられること。初期値は例えば50枚とする。
- 抽出できない画像参照があっても、記事全体の処理を停止しないこと。
- 画像抽出結果はJSONL形式で保存し、再実行・監査・差分更新に利用できること。

### 3.4 Commonsメタデータ取得機能

システムは、抽出した画像ファイル名をもとにWikimedia Commons APIを呼び出し、画像URLとメタデータを取得できること。

取得対象のAPIは以下とする。

```text
https://commons.wikimedia.org/w/api.php
?action=query
&prop=imageinfo
&titles=File:Example.jpg
&iiprop=url|size|mime|extmetadata
&iiurlwidth=768
&format=json
```

Commonsの `imageinfo` プロパティは、画像URL、サイズ、MIMEタイプ、拡張メタデータを取得できる。

取得する項目は以下とする。

| 項目 | 用途 |
|---|---|
| `thumburl` | 埋め込み・表示用のサムネイルURL |
| `url` | オリジナル画像URL |
| `mime` | JPEG / PNG / WebP / SVGなどの判別 |
| `width` | 画像幅 |
| `height` | 画像高さ |
| `descriptionurl` | Commonsのファイル説明ページURL |
| `LicenseShortName` | CC BY、CC BY-SA、Public domainなど |
| `Artist` | 作者情報 |
| `Credit` | クレジット情報 |
| `LicenseUrl` | ライセンスURL |
| `ImageDescription` | 画像説明 |
| `DateTimeOriginal` | 撮影日時、取得可能な場合 |

仕様は以下とする。

- 1リクエストあたり最大50ファイルをまとめて照会する。
- `iiurlwidth=768` を既定とし、埋め込み用にはサムネイルを使用する。
- オリジナル画像URLも保存するが、埋め込みには原則サムネイルを使用する。
- SVGはPNGへの変換が可能な場合のみ処理し、変換できない場合はスキップする。
- GIF、動画、音声、PDFなどは既定で除外する。
- API失敗時は指数バックオフで再試行する。
- レート制限を避けるため、同時実行数を制御する。

### 3.5 画像ダウンロード機能

システムは、Commons APIから取得した画像URLから画像ファイルをダウンロードできること。

仕様は以下とする。

- 既定では幅768px程度のサムネイルをダウンロードする。
- 画像はローカルディスクまたはオブジェクトストレージに保存する。
- ファイル名は、`File:`タイトルのSHA-1ハッシュをベースにする。
- 同一ファイルは再ダウンロードしない。
- ダウンロード済み画像には、ファイルサイズ・ハッシュ・取得日時を記録する。
- 破損画像、0バイトファイル、HTMLエラーページ、極端に小さい画像は除外する。
- 画像の最小サイズは、例えば幅・高さともに100px以上とする。
- 画像の最大サイズは、例えば幅・高さともに8,192px以下とする。
- ダウンロードの同時実行数は、既定で2〜4程度に制限する。
- 失敗した画像は、失敗理由付きでエラーログに記録する。

### 3.6 画像前処理機能

システムは、埋め込み前に画像を検証・正規化できること。

- JPEG、PNG、WebPを対象とする。
- SVGは、PNGへ変換できる場合のみ対象とする。
- 埋め込み用にはRGBへ変換する。
- アニメーションGIFは除外する。
- 透過PNGは、必要に応じて白背景へ合成する。
- 極端に縦長・横長の画像は、モデルの入力仕様に合わせてリサイズする。
- 画像が読み込めない、破損している、または極端に小さい場合はスキップする。
- 処理済み画像は、元画像と処理済み画像を区別して保存する。

### 3.7 SigLIP2埋め込み機能

システムは、処理済み画像を `google/siglip2-so400m-patch16-384` で埋め込みできること。

Hugging Faceの当該モデルは、`AutoModel` と `AutoProcessor` を用いて画像をエンコードし、`get_image_features()` で画像特徴量を取得する方式が提供されている。 [huggingface](https://huggingface.co/google/siglip2-so400m-patch16-384)

仕様は以下とする。

- モデルIDは `google/siglip2-so400m-patch16-384` とする。
- 入力画像はRGB形式とする。
- 埋め込みベクトルは、モデル出力後にL2正規化する。
- ベクトルは `float32` で保存する。
- バッチ処理に対応すること。
- GPUがある場合はGPUを使用し、ない場合はCPUで処理する。
- 埋め込み失敗時は、該当画像のみをスキップし、全体処理を継続する。
- 埋め込み済み画像は再計算しない。
- モデル名、モデルリビジョン、transformersバージョンを記録する。
- 実際のベクトル次元は、モデル初期化後に確認してZilliz Cloudのコレクション作成に反映すること。

### 3.8 Zilliz Cloud登録機能

システムは、画像埋め込みと関連メタデータをZilliz Cloudへ登録できること。

Zilliz CloudはPyMilvus経由でコレクション作成、スキーマ定義、ベクトル登録が可能である。

登録単位は **画像1枚につき1レコード** とする。

| フィールド | 型 | 必須 | 内容 |
|---|---|---|---|
| `id` | INT64、主キー | 必須 | 画像レコードの一意ID |
| `vector` | FLOAT_VECTOR | 必須 | SigLIP2画像埋め込み |
| `page_id` | INT64 | 必須 | Wikivoyage記事ID |
| `page_title` | VARCHAR | 必須 | Wikivoyage記事タイトル |
| `page_url` | VARCHAR | 必須 | Wikivoyage記事URL |
| `image_title` | VARCHAR | 必須 | Commonsの `File:...` 名 |
| `image_url` | VARCHAR | 必須 | ダウンロード・表示用画像URL |
| `image_page_url` | VARCHAR | 必須 | CommonsファイルページURL |
| `caption` | VARCHAR | 任意 | 画像キャプション |
| `license` | VARCHAR | 必須 | ライセンス名 |
| `artist` | VARCHAR | 任意 | 作者・帰属情報 |
| `credit` | VARCHAR | 任意 | クレジット情報 |
| `mime_type` | VARCHAR | 必須 | `image/jpeg` など |
| `width` | INT64 | 任意 | 元画像の幅 |
| `height` | INT64 | 任意 | 元画像の高さ |
| `dump_date` | VARCHAR | 必須 | ダンプ日付 |
| `embedding_model` | VARCHAR | 必須 | `google/siglip2-so400m-patch16-384` |
| `created_at` | INT64 | 必須 | レコード作成時刻 |

`page_url` は次の形式で生成する。

```text
https://en.wikivoyage.org/wiki/<URL-encoded_page_title>
```

例：

```text
https://en.wikivoyage.org/wiki/Tokyo
```

### 3.9 コレクション作成機能

システムは、Zilliz Cloudにコレクションが存在しない場合のみ作成すること。

- コレクション名は環境変数または設定ファイルで指定する。
- 既定のコレクション名は `wikivoyage_images` とする。
- ベクトル次元は、SigLIP2の出力次元を実行時に確認して設定する。
- 類似度指標は `COSINE` を採用する。
- インデックスはZilliz Cloudの `AUTOINDEX` を利用する。
- 主キーは `id` とする。
- 動的フィールドは無効にする。
- 既存コレクションがある場合は、スキーマ互換性を確認してから登録する。
- スキーマが異なる場合は、誤って上書きせずエラーを通知する。

Zilliz Cloudのスキーマでは、主キー、文字列フィールド、整数フィールド、`FLOAT_VECTOR` フィールドを定義でき、ベクトル次元を指定できる。

### 3.10 バッチ登録機能

システムは、Zilliz Cloudへの登録をバッチ単位で行うこと。

- 既定のバッチサイズは100レコードとする。
- バッチ登録前に、必須項目・ベクトル次元・データ型を検証する。
- 登録成功時は、登録件数と処理時間を記録する。
- 登録失敗時は、失敗したバッチをJSONLへ退避する。
- 退避データは再投入可能であること。
- 同一画像の再登録を避けるため、`id` による重複排除を行うこと。
- 処理中断後も、処理済み画像を再計算せず再開できること。

## 4. 非機能要件

| 項目 | 要件 |
|---|---|
| 可用性 | ダンプ解析・画像取得・埋め込み・登録を独立して再実行できること |
| 冪等性 | 同一ダンプ・同一画像の再実行で重複登録や重複計算を避けること |
| 処理方式 | ストリーム処理を基本とし、ダンプ全体をメモリに保持しないこと |
| 並列処理 | Commons API・画像ダウンロード・埋め込みを制御付き並列で実行すること |
| レート制限 | WikimediaのUser-Agentポリシー、接続数制限、レート制限を遵守すること |
| ログ | 処理開始・終了、ページ数、画像数、成功・失敗件数、エラー内容を記録すること |
| 監査性 | どのダンプ・ページ・画像からベクトルが作られたかを追跡できること |
| 再現性 | モデル名、モデルバージョン、ダンプ日付、処理日時を記録すること |
| 拡張性 | 日本語版Wikivoyage、Wikipedia、他言語版への拡張を考慮すること |
| セキュリティ | Zilliz Cloudの接続情報、APIキー、Hugging Faceトークンをコードにハードコードしないこと |

## 5. データフロー

```text
英語Wikivoyageダンプ
        |
        v
XMLストリーム解析
        |
        v
記事ページ抽出
        |
        v
wikitextから全画像参照抽出
        |
        v
Wikimedia Commons API
        |
        +--> 画像URL
        +--> ライセンス
        +--> 作者
        +--> MIMEタイプ
        |
        v
画像ダウンロード
        |
        v
画像検証・前処理
        |
        v
SigLIP2埋め込み
        |
        v
Zilliz Cloud登録
        |
        +--> page_url
        +--> page_title
        +--> image_url
        +--> license
        +--> vector
```

## 6. 入出力仕様

### 6.1 入力

| 入力 | 内容 |
|---|---|
| ダンプ日付 | 例：`20261101` |
| ダンプURL | 例：`https://dumps.wikimedia.org/enwikivoyage/20261101/enwikivoyage-20261101-pages-articles.xml.bz2` |
| Zilliz URI | Zilliz Cloudクラスタの接続URI |
| Zillizトークン | 接続認証情報 |
| コレクション名 | 例：`wikivoyage_images` |
| Hugging Faceトークン | モデル取得に必要な場合 |
| サムネイル幅 | 既定：768px |
| 画像上限 | 既定：記事あたり50枚 |
| バッチサイズ | 既定：100レコード |

### 6.2 出力

| 出力 | 内容 |
|---|---|
| Zilliz Cloudレコード | 画像埋め込みとメタデータ |
| 画像抽出JSONL | ページID、タイトル、ファイル名、キャプション |
| 画像メタデータJSONL | URL、ライセンス、作者、サイズ、MIMEタイプ |
| ダウンロードログ | 成功・失敗・スキップ理由 |
| 埋め込みログ | モデル、次元、処理時間、失敗画像 |
| Zilliz登録ログ | 登録件数、失敗バッチ、再投入ファイル |

## 7. エラー処理仕様

| 状況 | 挙動 |
|---|---|
| ダンプURLが無効 | 処理を停止し、URLと日付をエラー通知 |
| ダンプのチェックサム不一致 | ダンプを削除し、再ダウンロードを促す |
| XML解析エラー | 該当ページをスキップし、ログに記録 |
| 画像参照の抽出失敗 | 該当画像のみスキップ |
| Commons APIのタイムアウト | 指数バックオフで再試行 |
| 画像URLが存在しない | スキップし、失敗理由を記録 |
| 画像ダウンロード失敗 | 再試行後、失敗画像として記録 |
| 画像が破損 | スキップし、失敗理由を記録 |
| 埋め込み失敗 | 該当画像のみスキップ |
| Zilliz登録失敗 | バッチをJSONLへ退避し、再投入可能にする |
| Zillizスキーマ不一致 | 処理を停止し、既存コレクションの確認を促す |

## 8. ライセンス・帰属要件

画像はWikimedia Commons由来であり、ライセンスはファイルごとに異なる。システムは、少なくとも以下をZilliz Cloudレコードおよびローカル監査ファイルに保存すること。

- `license`
- `artist`
- `credit`
- `license_url`
- `image_page_url`
- `source_page_id`
- `source_page_title`
- `dump_date`

公開サービス・検索UI・商用利用で画像を表示する場合は、ライセンスごとの帰属表示、作者表記、ライセンスリンクを表示できる設計にする必要がある。Commons APIの `extmetadata` はライセンスや作者などの拡張メタデータを提供する。

## 9. 検索仕様

Zilliz Cloudに登録後、以下の検索を提供できること。

| 検索方式 | 入力 | 出力 |
|---|---|---|
| 画像類似検索 | ユーザー画像 | 類似画像、記事タイトル、記事URL、画像URL |
| 画像URL検索 | 画像URL | 類似画像、記事タイトル、記事URL |
| キャプション検索 | 任意テキスト | 類似画像、記事タイトル、記事URL |
| 記事フィルタ検索 | テキストまたは画像 + `page_title` | フィルタ後の類似画像 |

検索結果には、少なくとも以下を返すこと。

- `page_title`
- `page_url`
- `image_url`
- `image_page_url`
- `caption`
- `license`
- `artist`
- `score`

Zilliz Cloudの `search()` は、検索ベクトル、取得件数、返却フィールドを指定して類似検索を行える。

## 10. 初期リリースの推奨設定

| 項目 | 推奨値 |
|---|---|
| 対象Wiki | `enwikivoyage` |
| ダンプ | 現行版 `pages-articles` |
| 対象ページ | 名前空間0、リダイレクト除外 |
| 対象画像 | 記事内の全画像 |
| 記事あたり画像上限 | 50枚 |
| サムネイル幅 | 768px |
| 埋め込みモデル | `google/siglip2-so400m-patch16-384` |
| 類似度 | COSINE |
| バッチサイズ | 100 |
| ダウンロード並列数 | 3 |
| API同時実行数 | 2〜4 |
| 画像保存先 | ローカルSSDまたはS3/GCS |
| Zillizコレクション | `wikivoyage_images` |

## 11. 受け入れ基準

システムは、以下を満たすこととする。

- 指定した英語Wikivoyageダンプを取得できる。
- ダンプから名前空間0の記事を抽出できる。
- 記事内の全画像参照を抽出できる。
- Commons APIから画像URL、ライセンス、作者情報を取得できる。
- 画像をダウンロードし、破損・重複・不適切な形式を除外できる。
- `google/siglip2-so400m-patch16-384` で画像埋め込みを作成できる。
- Zilliz Cloudに、画像ベクトル、ページURL、ページタイトル、画像URL、ライセンスを登録できる。
- 同一ダンプ・同一画像の再実行で重複登録しない。
- 失敗した画像・失敗したバッチをログと退避ファイルで追跡できる。
- 画像類似検索により、記事タイトル・記事URL・画像URLを返せる。 
