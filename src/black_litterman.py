import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class black_litterman:
    def __init__(self, delta=2.5, tau=0.1625):
        self.delta = delta
        self.tau = tau
        pass

    def read_data(self):
        ## 時価総額ウェイトを求める
        self.market_weight_df = pd.read_csv(
            "./data/intermediate/market_weight.csv", index_col=0
        )
        self.market_weight_df.sort_values(by="Name", inplace=True)
        self.market_weight_array = np.array(self.market_weight_df["weight"]).reshape(
            [-1, 1]
        )

        # 共分散行列を読み込む
        self.Sigma_df = pd.read_csv("./data/intermediate/cov_mat.csv", index_col=0)
        self.Sigma_array = np.array(self.Sigma_df)

        # 投資家のビューを読み込む
        self.view_P_df = pd.read_csv("./data/input/P_mat.csv", index_col=0)
        self.view_P_array = np.array(self.view_P_df)

        self.view_Q_df = pd.read_csv("./data/input/Q_mat.csv", index_col=0)
        self.view_Q_array = np.array(self.view_Q_df).reshape([-1, 1])

    def calculate_implied_return(self):
        # 均衡リターンを求める(reverse optimization)
        self.implied_return_array = (
            self.delta * self.Sigma_array @ self.market_weight_array
        )
        self.implied_return_array = self.implied_return_array.reshape(-1, 1)

    def update_by_view(self):
        """
        均衡リターンに投資家のビューをブレンドし、事後リターン、事後分散を求める
        """
        # ビューの自身度合いを表す\Omegaを求める
        Omega_list = []
        for i in range(len(self.view_Q_df)):
            Omega_list.append(
                float(self.view_P_array[i] @ self.Sigma_array @ self.view_P_array[i].T)
            )

        self.Omega_array = np.diag(Omega_list)

        # 事後リターンを求める
        self.posterior_return = (
            self.implied_return_array
            + self.tau
            * self.Sigma_array
            @ self.view_P_array.T
            @ np.linalg.inv(
                self.tau * self.view_P_array @ self.Sigma_array @ self.view_P_array.T
                + self.Omega_array
            )
            @ (self.view_Q_array - self.view_P_array @ self.implied_return_array)
        )

        # 事後共分散行列を求める
        self.posterior_Sigma = (
            self.Sigma_array
            + self.tau * self.Sigma_array
            - self.tau
            * self.Sigma_array
            @ self.view_P_array.T
            @ np.linalg.inv(
                self.tau * (self.view_P_array @ self.Sigma_array @ self.view_P_array.T)
                + self.Omega_array
            )
            @ (self.tau * (self.view_P_array @ self.Sigma_array))
        )

        # 事後ウェイトを求める
        self.posterior_weight_array = (
            np.linalg.inv(self.delta * self.posterior_Sigma) @ self.posterior_return
        )
        self.posterior_weight_array /= sum(self.posterior_weight_array)

    def output_result(self):
        """
        結果を出力する
        """
        # 重み
        self.posterior_weight_df = self.market_weight_df.copy()
        self.posterior_weight_df["weight"] = self.posterior_weight_array
        self.posterior_weight_df.to_csv("./data/output/result_weight.csv")


if __name__ == "__main__":
    ins = black_litterman()
    ins.read_data()
    ins.calculate_implied_return()
    ins.update_by_view()
    ins.output_result()
