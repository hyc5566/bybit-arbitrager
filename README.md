# Bybit 期現套利程式

這支程式是用來作期現貨套利賺取資金費率下單使用的。使用 pybit 這個套件來運作 Bybit 的 API 以詢價、期現貨快速下單。

## 安裝

下載
```
git clone https://github.com/hyc5566/bybit-arbitrager
```
進入到目錄底下安裝必要的套件
```
cd bybit-arbitrager
pip install -r requirements.txt
```

## 申請 API key

直接到 Bybit 登入後申請 API key，並將 key 跟 secret 填在 `spot-contract-arbitrager.py` 裡面。
為了資產安全問題，請開設子帳號並挪移部分資產至子帳號使用。

## 使用

在 candidate_coins.txt 增加想做套利的幣種，直接執行 `pyhton spot-contract-arbitrager.py`。

## 剩下的未完成功能

- 初始保證金計算公式不太準確，無法正確估計多、空的最大開倉數量
- 重複使用到 API 的功能仍須優化
- 其他每次新增多餘的 bug...


