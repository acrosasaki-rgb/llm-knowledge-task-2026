# AKBC Shared Task 2026 システム論文

## レギュレーション（2026-08-11 に https://lm-kbc.github.io/challenge2026/ で確認）

| 項目 | 内容 |
|---|---|
| フォーマット | **ACLスタイル・2カラム・本文最大6ページ**（`acl.sty` 同梱、acl-org/acl-style-files 由来） |
| 提出先 | OpenReview: `EMNLP/2026/Workshop/LM-KBC_Shared_Task` |
| 査読 | Single-blind（著者名は記載してよい） |
| **提出締切** | **2026-08-15**（テストデータ公開 08-08、採否通知 09-01、camera-ready 09-15、発表は10月ブダペスト） |
| 提出物 | ①Codabench への予測提出 ②システム論文 ③**公開GitHubリポジトリ（論文にリンク必須）** |
| モデル制約 | 推論時の全ニューラル成分合計 **32Bパラメータ以下**。量子化してもカウントは減らない。MoEは総パラメータ数 |
| Closed-book | Web検索・RAG・外部コーパス・KB参照禁止。**追加学習（FT/継続事前学習）禁止**。エージェント的多段推論と非ニューラル処理（正規化・集約等）は可 |

## ファイル

- `main.tex` — 論文骨子。各節にTODOコメントで書く内容と既知の数値を記載済み
- `references.bib` — 主要文献の雛形（**要検証**、特にLM-KBC過去版のoverview論文）
- `acl.sty` / `acl_natbib.bst` — 公式ACLスタイル

## ビルド

```bash
latexmk -pdf main.tex
```

Overleaf の場合は `paper/` の4ファイルをアップロードすればそのまま通る。

## 提出前チェックリスト

- [ ] 本文6ページ以内（Limitations・references・appendix の扱いを OpenReview の CFP で確認）
- [ ] test 予測を Codabench に提出し、スコアを Table 1 に反映
- [ ] リポジトリを public 化し URL を Appendix に記載
- [ ] BibTeX の TODO を解消
- [ ] 単一seed・小サンプル（awardWonBy val 10行）の注意書きを Limitations に明記
