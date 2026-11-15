# StructArray スパイク結果（設計書 `structarray-db-design.md` §10-1）

- 実施日: 2026-11-14
- スクリプト: `scripts/spike_structarray.py`（生結果 `articles_work/spike.json`）
- クラスタ: `.env` の `ZILLIZ_URI`。`get_server_version()` = **"Zilliz Cloud Vector Database (Compatible with Milvus 3.0)"**
- クライアント: pymilvus 3.0.0
- 使い捨てのコレクション `wikivoyage_articles_spike*` で検証し、終了後に削除済み。既存の `wikivoyage_pages` と `conversation_chunks_bm25_v2` には触れていない。

テストのスキーマは設計どおりとした。PK `page_id`、トップレベル `article_embedding`(1024)、`chunks` StructArray(cap 512; chunk_id, chunk_index, top_section, text, embedding 1024)、`images` StructArray(cap 64; file_title, caption, position, embedding 1152)。ベクトルは乱数を L2 正規化した実次元のもの。

## 結果一覧

| # | 確認項目 | 結果 | 詳細 |
|---|---|---|---|
| a | StructArray コレクションの作成（StructArray 2 つ、HNSW/COSINE） | **OK** | 現クラスタで作成できた |
| b | トップレベルのベクトルフィールドが必須か | **不要** | StructArray のベクトルだけでも作成・insert できた |
| c | `max_capacity` 512 | スキーマ **OK** / 512 件の行は **NG** | 下記「1 行 2 MiB 上限」を参照 |
| c2 | 513 件（容量超過）の insert | 期待どおり拒否 | `array length exceeds max capacity` |
| d | 空配列 `images=[]` の insert | **OK** | query では `[]` で返る。`chunks=[]` も insert できた |
| e | StructArray を含む行の `upsert`（全体置換） | **OK** | chunks 数が置換後の 2 件になり、内容も更新された |
| e2 | `partial_update=True` | **NG** | `Struct fields are not supported in partial update`（pymilvus 側のエラー） |
| f | 要素レベル検索（`chunks[embedding]`、COSINE） | **OK** | 各ヒットに `offset`（一致した要素の添字）が付く。PK はヒットの `page_id` キーに入り、`id` は None |
| f' | `output_fields=["chunks[text]"]` | **配列全体が返る** | 一致した要素だけではなく全要素の `text` が返る。指定したサブフィールドだけに絞られる |
| f2 | 同じ記事の複数要素がヒットするか | **する** | 1 記事が複数ヒットになるので、クライアント側で記事単位にまとめる必要がある |
| f3 | 画像要素の検索（`images=[]` の記事を含む） | **OK** | 空配列の記事が混ざっても問題なし |
| g | 記事条件 + `element_filter(chunks, $[top_section] == "See")` | **OK** | `page_type == "city" && element_filter(...)` が効き、ヒットした offset は全て See |
| g2 | `MATCH_ANY` / `array_contains(chunks[top_section], …)` / `array_length(images) == 0` | **OK** | いずれも期待どおりの記事集合を返した |
| h | サブフィールドに AUTOINDEX（ベクトル）+ INVERTED（スカラー） | **OK** | HNSW 指定でも作成できた |
| i | 1 リクエストのサイズ（約 2 / 8 / 32 MB） | **OK** | 200 チャンク×2000 文字の大型行で 2/7/26 行。所要 4/11/45 秒 |
| j | `hybrid_search`（`chunks[embedding]` + `images[embedding]`、RRF） | 動作は **OK** | 記事単位の結果になり `offset` は返らない（None） |
| x | `delete(filter="page_id in [...]")` | **OK** | replace モードの代替手段として使える |
| k | pymilvus の最小バージョン | 3.0.0 で動作 | `create_struct_field_schema`、ヒットの `offset` がある |

## 重要な発見: 1 行 2 MiB 上限（WAL メッセージ）

```
code=1102 parameter too large: single row at offset 0 is too large to fit in one WAL message:
estimated size=2466487 bytes, limit=2097152 bytes
```

- **1 エンティティ ≤ 2,097,152 バイト**。上限はリクエスト単位ではなく行単位で、リクエストは 32 MB でも通る。
- 見積もりサイズは、生データのバイト数（ベクトル 4 B×dim + 文字列長）にほぼ等しい。512 チャンク×(4096+約110 B) + 64 画像×約4.7 KB ≒ 2.47 MB で拒否された。
- 実データ（`chunks_embedded/`、25,281 記事）でテキスト側の行サイズを見積もった。

  | | テキスト側の行サイズ見積もり |
  |---|---|
  | 中央値 | 46 KB |
  | p99 | 340 KB |
  | 最大 | 1.10 MB（page_id 2001、206 チャンク） |
  | 1 MB 超 | 1 記事 |

  画像が上限の 64 枚あるとさらに約 0.3〜0.45 MB 増え、最大でも約 1.55 MB。現データは 2 MiB に収まる。ただし余裕は約 25% しかない。

## 設計への反映が必要な点

1. **行サイズの上限チェックを assemble に追加する**：`row_bytes ≤ 1.9 MiB` を検証し、超えたら画像 → 末尾チャンクの順で削って `overflow.jsonl` に記録する。`max_capacity` 512 は、2 MiB 上限があるため実質的には到達しない。容量 512 はそのまま使える。
2. **書き込みモードは `upsert`（全体置換）に確定**：delete+insert は不要。partial update は StructArray を含むと使えないため、記事メタだけの更新も行全体を送る。
3. **バッチ上限**：1 リクエストは件数 50 / 8 MB の現設計でよい（32 MB でも成功）。
4. **検索応答**：`output_fields` に `chunks[text]` を指定すると記事の全チャンク本文が返る（最大約 0.4 MB/記事）。次のどちらかを選ぶ。
   - (A) search では `page_id` と `offset` だけを取り、上位の記事について `query(output_fields=["chunks"])` で取り出し、offset で切り出す（設計 §7.3 の 2 段階方式）
   - (B) limit が小さい（≤10）ならそのまま受け取り、offset で切り出す

   →(A) を推奨。
5. **空配列のプレースホルダは不要**：`images=[]` も `chunks=[]` も insert・検索できた（§8 のフォールバックは削除してよい）。
6. **`article_embedding` は任意**：トップレベルのベクトルは必須ではない。記事単位の検索に使うなら残す。
7. **テキストと画像の統合はクライアント側の RRF のままとする**：`hybrid_search` は動くが `offset` が返らないので、どの要素が一致したかが分からない。
8. **前提の訂正**：設計書は On-Demand/Serverless を前提にしていたが、現クラスタは Milvus 3.0 互換と表示され、StructArray が使えることを確認した。クラスタの種別（Dedicated か Serverless か）は API からは判別していない。

## 補足

- 現クラスタにはまだ `wikivoyage_images` コレクションがない（画像パイプラインの upsert は未実行）。本設計は画像パイプラインのファイル出力（`embeddings/…parquet` など）を読むため影響はない。ただし `embed` までは実行しておく必要がある。
- 上記はテスト用の乱数ベクトルでの動作確認であり、検索精度は検証していない。
