- input
    - universe.xlsx
    投資対象とするユニバースを定義するためのリスト
    coincheckから購入可能な全20銘柄

- intermediate
    - price_data.csv
    coingecho APIから取得した価格データ
    universe.xlsxで指定した銘柄のみ取得する

    - Market_Cap_ranking.csv
    時価総額データ（上位100位のみ）

    - cov_mat.csv
    日次対数収益率の共分散行列
    ただし、時価総額上位100位を対象

- output
    - 