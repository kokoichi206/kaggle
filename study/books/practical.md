## memo

- コンペの内容の理解

## data

順序尺度（数値の順番に意味があるもの）と、間隔尺度（間隔に意味があるもの）とかで、区別をつける。

順序尺度のみの場合、もしかしたら「カテゴリ変数」と捉えた方が、数値は良くなるかもしれない。

```python
df_train["Pclass"] = df_train["Pclass"].astype(object)
```

欠損値の確認

```python
df_train.isnull().sum()
```

## ベースラインの作成

ベースライン作成時には、数個のデータ項目のみを使う。
問題となりそうなデータは最初は除いてスタートする。

```python
x_train, y_train, id_train = df_train[["Pclass", "Fare"]], \
                                df_train[["Survived"]], \
                                df_train[["PassengerId"]]
print(x_train.shape, y_train.shape, id_train.shape)
```

バリデーションの目的は、作成するモデルの精度を手元のデータで判断すること！

検証データは「実際の適用シーンと同じ状況を仮想的に再現する」ことが理想。

### ホールドアウト検証

```python
x_tr, x_va, y_tr, y_va = train_test_split(x_train,
                                          y_train,
                                          test_size=0.2,
                                          shuffle=True,
                                          # 目的変数の偏りを均等に分配する
                                          stratify=y_train,
                                          random_state=123)
print(x_tr.shape, y_tr.shape)
print(x_va.shape, y_va.shape)
print("y_train: {:.3f}, y_tr: {:.3f}, y_va:{:.3f}".format(
    y_train["Survived"].mean(),
    y_tr["Survived"].mean(),
    y_va["Survived"].mean(),
))
```

### クロスバリデーション

目的変数の「0」と「1」の割合を揃えたい時などは、StratifiedKFold が使える。

```python
n_splits = 5
cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(x_train, y_train))

for nfold in np.arange(n_splits):
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
    x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
    x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]
    print(x_tr.shape, y_tr.shape)
    print(x_va.shape, y_va.shape)
    print("y_train: {:.3f}, y_tr: {:.3f}, y_va:{:.3f}".format(
        y_train["Survived"].mean(),
        y_tr["Survived"].mean(),
        y_va["Survived"].mean(),
    ))
```

### モデル学習（LightGBM）

Gradient Boosting Machine

テーブルデータに対するベースラインは、基本「LightGBM」でよい。

- 精度が高い
- 処理が高速
- カテゴリ変数を数値に変換しなくていい
- 欠損値があっても処理可能
- 異常値の影響を受けにくい

ホールドアウト

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'n_estimators': 100000,  # ?
    'random_state': 123,
    'importance_type': 'gain',  # ?
}

model = lgb.LGBMClassifier(**params)
model.fit(x_tr,
         y_tr,
         eval_set=[(x_tr, y_tr), (x_va, y_va)],
         early_stopping_rounds=100,
         verbose=10,  # ?
         )

y_tr_pred = model.predict(x_tr)
y_va_pred = model.predict(x_va)
metric_tr = accuracy_score(y_tr, y_tr_pred)
metric_va = accuracy_score(y_va, y_va_pred)
print("[accyracy] tr: {:.2f}, va: {:.2f}".format(metric_tr, metric_va))

# 説明変数の中で何の寄与が大きいか
imp = pd.DataFrame({"col": x_train.columns, "imp": model.feature_importances_})
imp.sort_values("imp", ascending=False, ignore_index=True)
```

CV

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'n_estimators': 100000,  # ?
    'random_state': 123,
    'importance_type': 'gain',  # ?
}

metrics = []
imp = pd.DataFrame()

n_splits = 5
cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(x_train, y_train))

for nfold in np.arange(n_splits):
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
    x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
    x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]
    print(x_tr.shape, y_tr.shape)
    print(x_va.shape, y_va.shape)
    print("y_train: {:.3f}, y_tr: {:.3f}, y_va:{:.3f}".format(
        y_train["Survived"].mean(),
        y_tr["Survived"].mean(),
        y_va["Survived"].mean(),
    ))

    model = lgb.LGBMClassifier(**params)
    model.fit(x_tr,
             y_tr,
             eval_set=[(x_tr, y_tr), (x_va, y_va)],
             early_stopping_rounds=100,
             verbose=100,  # ?
             )

    y_tr_pred = model.predict(x_tr)
    y_va_pred = model.predict(x_va)
    metric_tr = accuracy_score(y_tr, y_tr_pred)
    metric_va = accuracy_score(y_va, y_va_pred)
    print("accuracy tr: {:.2f}, va: {:.2f}".format(metric_tr, metric_va))
    metrics.append([nfold, metric_tr, metric_va])

    _imp = pd.DataFrame({"col": x_train.columns, "imp": model.feature_importances_, "nfold": nfold})
    imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

print("-------------- result --------------")
print(metrics)
metrics = np.array(metrics)
print(metrics)
print(metrics[:,1].mean, metrics[:,1].std())
print(metrics[:,2].mean, metrics[:,2].std())

imp = imp.groupby("col")["imp"].agg(["mean", "std"])
imp.columns = ["imp", "imp_std"]
print(type(imp))
imp = imp.reset_index(drop=False)

imp.sort_values("imp", ascending=False, ignore_index=True)
```

### ベースラインの正しさの検証

あらかじめデータセットから、ベースライン検証用データを切り出しておく。

Kaggle の output の話で、Kaggle では自然と検証できている。

```python
# ベースライン確認用のデータを分割
x_tr, x_va2,y_tr, y_va2 = train_test_split(x_train,
                                          y_train,
                                          test_size=0.2,
                                          shuffle=True,
                                          stratify=y_train,
                                          random_state=123)
print(x_tr.shape, y_tr.shape)
print(x_va2.shape, y_va2.shape)

# 残りから、学習データと検証データを分割
x_tr1, x_va1,y_tr1, y_va1 = train_test_split(x_tr,
                                          y_tr,
                                          test_size=0.2,
                                          shuffle=True,
                                          stratify=y_tr,
                                          random_state=123)
print(x_tr1.shape, y_tr1.shape)
print(x_va1.shape, y_va1.shape)


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'n_estimators': 100000,  # ?
    'random_state': 123,
    'importance_type': 'gain',  # ?
}
model = lgb.LGBMClassifier(**params)
model.fit(x_tr1,
         y_tr1,
         eval_set=[(x_tr1,y_tr1), (x_va1,y_va1)],
         early_stopping_rounds=100,
         verbose=10)

y_va1_pred = model.predict(x_va1)
y_va2_pred = model.predict(x_va2)

print("検証データ: acc: {:.4f}".format(accuracy_score(y_va1, y_va1_pred)))
print("ベースライン検証データ: acc: {:.4f}".format(accuracy_score(y_va2, y_va2_pred)))
```

### データのプロット

```python
%matplotlib inline

y_va1_pred_prob = model.predict_proba(x_va1)[:,1]
y_va2_pred_prob = model.predict_proba(x_va2)[:,1]

fig = plt.figure(figsize=(10, 8))

fig.add_subplot(2,1,1)
plt.title("val data")
plt.hist(y_va1_pred_prob[np.array(y_va1).reshape(-1)==1], bins=10, alpha=0.5, label="1")
plt.hist(y_va1_pred_prob[np.array(y_va1).reshape(-1)==0], bins=10, alpha=0.5, label="0")
plt.grid()
plt.legend()

fig.add_subplot(2,1,2)
plt.title("base val data")
plt.hist(y_va2_pred_prob[np.array(y_va2).reshape(-1)==1], bins=10, alpha=0.5, label="1")
plt.hist(y_va2_pred_prob[np.array(y_va2).reshape(-1)==0], bins=10, alpha=0.5, label="0")
plt.grid()
plt.legend()
```

### 推論

```python
df_test = pd.read_csv("../input/titanic/test.csv")
x_test = df_test[["Pclass", "Fare"]]
id_test = df_test[["PassengerId"]]

# 学習済みモデルで推論
y_test_pred = model.predict(x_test)

# 提出フォーマットに合わせてデータを準備する
df_submit = pd.DataFrame({"PassengerId": id_test["PassengerId"], "Survived": y_test_pred})
display(df_submit.head(5))
df_submit.to_csv("submission_baseline.csv", index=None)
```

### 提出手順

1. Data > Output から、提出したデータをダウンロード
1. コンペのページに戻り、右上の Submit Predictions からファイルをアップロード

