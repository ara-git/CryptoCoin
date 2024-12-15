import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class black_litterman:
    def __init__(self, delta=2.5, tau=0.05):
        self.delta = delta
        self.tau = tau
        pass

    def read_data(self):
        ## 時価総額ウェイトを求める
        self.market_weight_df = pd.read_csv(
            "./data/intermediate/market_weight.csv", index_col=0
        )
        self.market_weight_df.sort_values(by="Name", inplace=True)
        self.market_weight_array = np.array(self.market_weight_df["weight"])

        # 共分散行列を読み込む
        self.cov_mat_df = pd.read_csv("./data/intermediate/cov_mat.csv", index_col=0)
        self.cov_mat_array = np.array(self.cov_mat_df)

    def calculate_implied_return(self):
        # 均衡リターンを求める(reverse optimization)
        self.implied_return_array = self.delta * np.dot(
            self.cov_mat_array, self.market_weight_array
        )


if __name__ == "__main__":
    ins = black_litterman()
    ins.read_data()
    ins.calculate_implied_return()

    print("##")
    print(ins.market_weight_df)
    print(ins.cov_mat_df)
    print(ins.implied_return_array)
