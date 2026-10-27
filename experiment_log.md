# Titanic ループエンジニアリング実験ログ

## 実験概要
- **データ**: C:\Users\kkoya\Desktop\ブログ\loopenngineering\train.csv / test.csv
- **目標**: Kaggle public score > 0.80
- **外側ループ上限**: 8回（= Kaggle 提出8回）
- **内側ループ**: CV が改善するまで手法を試し続け、改善したら Kaggle に提出

## 提出記録
| 提出# | 内側で試した手法数 | 採用アプローチ | CV | Kaggle | 判断 |
|-------|-----------------|--------------|-----|--------|------|

| 1 | 3手法比較(A:RF B:RF+SVC+LR C:GB) | RF+SVC+LR | 0.8339 | 0.78229 | gap=0.052。目標Kaggle 0.80にはCV>0.852が必要 |
| 2 | 5手法比較(D:RF+GB+SVC+LR=0.8294 E:SVC=0.8260 F:RF深め=0.8272 G:+TicketGroupSize=0.8339 H:RF500seed0=0.8316) | G: RF+SVC+LR+TicketGroupSize | 0.8339 | 0.78229 | CVは変わらず。TicketGroupSize追加も効果なし |
| 3 | 6手法比較(I:XGB=0.8070 J:XGB+full=0.8171 K:Stacking=0.8294 L:+domain=0.8350 M:XGB+domain=0.8171 N:Voting+domain=0.8305) | L: RF+SVC+LR + domain feats | 0.8350 | 0.77511 | CVは改善(+0.0011)だがKaggleは悪化。gap拡大(0.060)→ドメイン特徴量過学習 |
| 4 | 12手法比較(O:LGB=0.8126 P:LGB+Cabin=0.8160 Q:+Cabin=0.8305 R:ET+SVC+LR=0.8294 S:RF+ET+SVC+LR=0.8305 T:RandSearch=0.8305 U:RF(d4)+SVC+LR=0.8316 V:XGB_tuned=0.8260 W:Voting+ext=0.8260) | baseline RF+SVC+LR | 0.8339 | 0.78229 | 12手法試したが0.8339を超えられずbaselineで提出 |
| 5 | 3手法比較(X:HGB=0.8081 Y:HGB+Cabin=0.8204 Z:RF+SVC+LR+FamSurvRate=0.8462) | Z: RF+SVC+LR + FamilySurvivalRate(LOO) | 0.8462 | **0.81339** | **目標達成！** gap=0.033。ファミリー生存率(LOO target encoding)が決定的 |

## 実験結果まとめ
- **5回目の提出でKaggle 0.80超え達成（0.81339）**
- **stop条件: Kaggle > 0.80 → ループ終了**

## 教訓メモ（エージェントが毎回更新）
- CV-Kaggle gap: Sub1/2=0.052、Sub3=0.060、Sub5=0.033
- **決定的な改善要因: FamilySurvivalRate（同姓ファミリーの生存率）のLOO target encoding**
- WomanOrChild/FarePerBin/Pclass_Sexなどドメイン特徴量はCV改善するが過学習しやすい
- XGBoost/LightGBM単体はRF+SVC+LR ensembleより低い（891行という少データでは）
- **CVが高くてもKaggle悪化するケースあり → CVの微増より汎化性重視が必要**
- LOO target encodingは正しく実装すればCVとKaggleの両方を改善できる
- HistGradientBoosting単体はbase featsでCV=0.8081と低い（この問題には不向き）
