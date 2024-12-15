import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt


class process_price_data:
    def __init__(self):
        pass

    def read_data(self):
        # 価格データ
        self.price_df = pd.read_csv("./data/intermediate/price_data.csv", index_col=0)
        # 時価総額データ
        self.Market_cap_df = pd.read_csv("./data/intermediate/Market_Cap_ranking.csv")

    def cleansing_data(self):
        """
        データを綺麗にする
        """
        # 価格データ
        self.price_df["timestamp"] = pd.to_datetime(self.price_df["timestamp"])
        self.price_df.set_index("timestamp", inplace=True)

        # 対数収益率過程に変換する
        self.log_rt_df = np.log(self.price_df).diff()

        # 欠損値処理
        self.log_rt_df.dropna(how="all", inplace=True)

        # 列名を変更する（一時的処理）
        new_col_name_list = [
            col_name.replace("_price", "") for col_name in self.log_rt_df.columns
        ]
        self.log_rt_df.columns = new_col_name_list

    def filter_price_data(self):
        """
        時価総額上位のコインのみ、価格データを取り出す
        """
        self.log_rt_df = self.log_rt_df.loc[
            :, self.log_rt_df.columns.isin(self.Market_cap_df["Name"])
        ]
        # 名前順にソートする
        self.log_rt_df.sort_index(axis=1, inplace=True)

        # 逆に、ユニバースに存在する銘柄のみ時価総額データを抽出する
        self.Market_cap_df = self.Market_cap_df[
            self.Market_cap_df["Name"].isin(self.log_rt_df.columns)
        ]

    def calculate_statistics(self):
        """
        共分散行列含め、統計量を計算する
        """
        # 共分散行列
        self.cov_mat_df = self.log_rt_df.cov()
        # 相関係数行列
        self.corr_mat_df = self.log_rt_df.corr()

    def export_figs(self):
        """
        画像を出力する
        """
        seaborn.heatmap(self.corr_mat_df, cmap="Blues", annot=True, fmt=".1f")
        plt.savefig("./data/figs/corr_mat.png")

        #
        plt.clf()

        # 円グラフ作成用に、時価総額下位のコインはひとまとめにする
        self.Market_cap_df_for_fig = self._aggregate_Market_Cap_df(self.Market_cap_df)

        # FigureとAxesを作成
        fig, ax = plt.subplots(figsize=(8, 8))  # サイズを指定
        # 時価総額ウェイトの円グラフ
        ax.pie(
            self.Market_cap_df_for_fig["Market Cap (JPY)"],
            autopct="%.1f%%",
            startangle=90,
            counterclock=False,
            textprops={"fontsize": 15},
            pctdistance=0.8,
        )

        # 円グラフの位置を下に移動
        # ax.set_position([0.4, 0.2, 0.1, 0.6])  # [left, bottom, width, height]

        plt.legend(
            self.Market_cap_df_for_fig["Name"],
            fontsize=10,
            loc="upper center",
            ncol=len(self.Market_cap_df_for_fig["Name"]),
        )
        plt.title("Original Weight", fontsize=30)
        plt.savefig("./data/figs/original_weight.png")

    def _aggregate_Market_Cap_df(self, input):
        """
        円グラフ作成用に、時価総額下位のコインはひとまとめにする
        """
        # 総額計算
        total_market_cap = input["Market Cap (JPY)"].sum()
        input["Percentage"] = input["Market Cap (JPY)"] / total_market_cap * 100

        # 5%未満の銘柄を Others として集約
        threshold = 5  # 5%の閾値
        below_threshold = input[input["Percentage"] < threshold]
        above_threshold = input[input["Percentage"] >= threshold]

        # "Others" の作成
        others_row = pd.DataFrame(
            {
                "Name": ["Others"],
                "Market Cap (JPY)": [below_threshold["Market Cap (JPY)"].sum()],
                "Percentage": [below_threshold["Percentage"].sum()],
            }
        )

        # データの統合
        final_df = pd.concat([above_threshold, others_row], ignore_index=True)
        return final_df

    def calc_market_weight(self):
        self.Market_cap_df["weight"] = (
            self.Market_cap_df["Market Cap (JPY)"]
            / self.Market_cap_df["Market Cap (JPY)"].sum()
        )
        self.market_weight_df = self.Market_cap_df[["Name", "weight"]]

    def export_data(self):
        # 共分散行列
        self.cov_mat_df.to_csv("./data/intermediate/cov_mat.csv")
        # マーケットウエイト
        self.market_weight_df.sort_values(by="Name").reset_index(drop=True).to_csv(
            "./data/intermediate/market_weight.csv"
        )
        # 対数収益率
        self.log_rt_df.to_csv("./data/intermediate/log_return.csv")


if __name__ == "__main__":
    ins = process_price_data()
    ins.read_data()
    ins.cleansing_data()
    ins.filter_price_data()
    ins.calculate_statistics()
    ins.calc_market_weight()
    ins.export_figs()
    ins.export_data()

    print(ins.log_rt_df)
