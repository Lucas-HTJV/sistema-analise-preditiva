"""
Módulo de visualização de dados.
Responsável por gerar gráficos de dispersão, regressão linear e log-log.
"""

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, df, col_x, col_y):
        self.df = df.copy()
        self.col_x = col_x
        self.col_y = col_y

    def _calcular_r2(self, y_real, y_pred):
        """
        Calcula o coeficiente de determinação (R²)
        """
        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plotar(self, salvar=False):
        """
        Gera gráficos:
        - Dispersão
        - Regressão linear
        - Regressão log-log

        Parameters
        ----------
        salvar : bool
            Se True, salva os gráficos como imagens
        """

        x = self.df[self.col_x].values
        y = self.df[self.col_y].values

        # Ordena para plotar linha corretamente
        ordem = np.argsort(x)
        x_sorted = x[ordem]
        y_sorted = y[ordem]

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # -------------------------
        # 1. DISPERSÃO
        # -------------------------
        axs[0].scatter(x, y)
        axs[0].set_title("Dispersão")
        axs[0].set_xlabel(self.col_x)
        axs[0].set_ylabel(self.col_y)

        # 2. REGRESSÃO LINEAR

        coef = np.polyfit(x, y, 1)
        y_pred = np.poly1d(coef)(x)

        r2 = self._calcular_r2(y, y_pred)

        axs[1].scatter(x, y)
        axs[1].plot(x_sorted, np.poly1d(coef)(x_sorted))
        axs[1].set_title(f"Regressão Linear (R²={r2:.4f})")
        axs[1].set_xlabel(self.col_x)
        axs[1].set_ylabel(self.col_y)


        # 3. REGRESSÃO LOG-LOG

        # Filtra valores inválidos (<=0)
        mask = (x > 0) & (y > 0)
        log_x = np.log10(x[mask])
        log_y = np.log10(y[mask])

        coef_log = np.polyfit(log_x, log_y, 1)
        y_log_pred = np.poly1d(coef_log)(log_x)

        r2_log = self._calcular_r2(log_y, y_log_pred)

        # Ordenar para linha ficar correta
        ordem_log = np.argsort(log_x)
        log_x_sorted = log_x[ordem_log]

        axs[2].scatter(log_x, log_y)
        axs[2].plot(log_x_sorted, np.poly1d(coef_log)(log_x_sorted))
        axs[2].set_title(f"Log-Log (R²={r2_log:.4f})")
        axs[2].set_xlabel(f"log10({self.col_x})")
        axs[2].set_ylabel(f"log10({self.col_y})")

        plt.tight_layout()

        if salvar:
            fig.savefig("graficos.png", dpi=300)

        plt.show()

        return {
            "r2_linear": r2,
            "r2_log": r2_log,
            "coef_linear": coef,
            "coef_log": coef_log
        }