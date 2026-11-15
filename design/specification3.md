embed_chunks.py



テキストのチャンクはembed_chunks.pyに使用しているLLMをしようしてほしい。
はい。WikiVoyageのRAGで、1記事＝1エンティティ、記事内のチャンク＝StructArray要素という設計はZilliz CloudのStructArrayに適しています。チャンクごとのテキスト・埋め込み・見出し・ページ位置を1つの記事にまとめられ、チャンク単位のベクトル検索とフィルタが可能です。[milvus][docs.zilliz][milvus]
推奨データ設計
WikiVoyage記事を、親エンティティとして保存します。
article_id: "Tokyo"
title: "Tokyo"
url: "https://en.wikivoyage.org/wiki/Tokyo"
language: "en"
last_updated: "2026-11-01"
chunks: StructArray
  ├─ chunk_id
  ├─ text
  ├─ heading
  ├─ section_path
  ├─ chunk_index
  ├─ embedding
  └─ image_embedding  ※画像検索も行う場合
StructArrayは、1エンティティ内に順序付きの複数Structを持ち、各Structにスカラーとベクトルを含められます。公式の例も「多数のチャンクを含む1つのドキュメント」を想定しており、WikiVoyageの記事・チャンク構造と合致します。[milvus][milvus]
スキーマ例
フィールド
型
役割
article_id
VARCHAR
WikiVoyage記事の安定ID
title
VARCHAR
記事タイトル
url
VARCHAR
元記事URL
language
VARCHAR
en、jaなど
last_updated
INT64
更新日時のUnixタイム
chunks
ARRAY<STRUCT>
チャンク配列
chunks.chunk_id
VARCHAR
Tokyo#chunk_012
chunks.text
VARCHAR
検索・回答に使う本文
chunks.heading
VARCHAR
直近の見出し
chunks.section_path
VARCHAR
Understand > Districts > Shibuya
chunks.chunk_index
INT64
記事内の順序
chunks.embedding
FLOAT_VECTOR
英語テキスト埋め込み
chunks.image_embedding
FLOAT_VECTOR
画像検索用SigLIP 2埋め込み
StructArrayの各要素は同一スキーマで、スカラー・ベクトルの両方を持てます。ただし、Struct内にさらにStruct、JSON、ネストした配列を持つことはできません。[milvus][milvus][milvus]
なぜStructArrayが向くか
WikiVoyageでは、1記事に「Understand」「Get in」「See」「Eat」「Sleep」など多くのセクションがあります。StructArrayを使うと、記事単位で更新・削除できつつ、検索はチャンク単位で行えます。[milvus][milvus]
	•	記事単位の更新：記事を再取得したら、その記事のエンティティを削除して再挿入するだけで、古いチャンクが残らない
	•	チャンク単位の検索：chunks[embedding] に対してベクトル検索する
	•	セクション絞り込み：chunks[heading] == "Sleep" や chunks[chunk_index] >= 10 で絞り込む
	•	複数チャンクを1回答に使う：親記事の article_id、title、url をまとめて取得し、LLMへ渡す
	•	画像検索との併用：chunks[image_embedding] にSigLIP 2のベクトルを入れ、テキスト検索と併用できる
インデックス
StructArrayでは、ベクトル検索やスカラー絞り込みの前に、サブフィールド単位でインデックスを作る必要があります。インデックス対象は chunks[embedding]、chunks[text]、chunks[heading] のようなパスです。[milvus][milvus]
サブフィールド
インデックス
用途
chunks[embedding]
HNSW等のベクトルインデックス
英語テキストによる画像・文章検索
chunks[image_embedding]
HNSW等のベクトルインデックス
SigLIP 2画像検索
chunks[heading]
スカラーインデックス
セクション絞り込み
chunks[chunk_index]
スカラーインデックス
順序・範囲条件
chunks[text]
必要に応じて
キーワードフィルタ
すべてのベクトルフィールドにはインデックスが必須で、コレクション全体とStructArray内を合わせたベクトルフィールド数は10以下に抑える必要があります。[blog.milvus][milvus]
検索クエリ例
「東京で夜景を見られる場所」を検索する場合：
query: "best places to see the night view in Tokyo"
filter: chunks[heading] in ["See", "Understand", "Districts"]
vector field: chunks[embedding]
top_k: 20
その後、取得したチャンクを article_id でグループ化し、同じ記事から複数チャンクを渡さないよう重み付けします。回答生成には、チャンク本文、見出し、記事URL、更新日をLLMに渡します。
画像検索との併用
風景画像検索も行うなら、SigLIP 2で画像を埋め込み、chunks[image_embedding] に保存します。テキスト検索は chunks[embedding]、画像検索は chunks[image_embedding] と、検索対象ベクトルを分けます。[milvus]
英語クエリ → SigLIP 2 text embedding → chunks[image_embedding] 検索
文章クエリ → テキスト埋め込み → chunks[embedding] 検索
注意点
	•	チャンクを大きくしすぎない：1チャンクは、見出し単位または300〜800トークン程度が扱いやすいです
	•	メタデータを重複させない：title や url は親レベルに置き、チャンク側には chunk_id・heading・chunk_index を持たせる
	•	更新は記事単位で：article_id で削除後に再挿入すると整合性を保てます
	•	NULLの扱いに注意：StructArrayのサブフィールドで、NULL配列と値あり配列を同じエンティティに混在させないでください。[milvus][milvus]
	•	深いネストは不可：StructArrayの中にさらにStructArrayやJSONは置けません。[milvus][milvus]
実装方針
	1	WikiVoyage記事を取得し、見出しごとに分割
	2	各チャンクに chunk_id、heading、section_path、chunk_index を付与
	3	SigLIP 2で英語テキストを埋め込み、chunks[embedding] に保存
	4	画像がある場合はSigLIP 2で画像を埋め込み、chunks[image_embedding] に保存
	5	記事単位でZilliz Cloudに挿入
	6	chunks[embedding] と chunks[image_embedding] にベクトルインデックスを作成
	7	検索結果を article_id でグループ化し、Qwen2.5-VLなどで回答生成
この構成なら、「記事単位の管理」と「チャンク単位の高精度検索」を両立でき、WikiVoyageのRAGにStructArrayはかなり適しています。[milvus][docs.zilliz][milvus]

