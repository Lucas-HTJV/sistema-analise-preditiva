import numpy as np
from sklearn.linear_model import LinearRegression


class RegressionModel:
    def __init__(self, df, col_x, col_y):
        self.df = df
        self.col_x = col_x
        self.col_y = col_y
        self.model = LinearRegression()
        self.metricas = None

    def _validar(self):
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame vazio.")

        for col in [self.col_x, self.col_y]:
            if col not in self.df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada.")

    def treinar(self):
        """Treina regressão linear simples"""
        self._validar()

        X = self.df[[self.col_x]].values
        y = self.df[self.col_y].values

        self.model.fit(X, y)

        coef = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)
        r2 = float(self.model.score(X, y))

        self.metricas = {
            "coeficiente": coef,
            "intercepto": intercept,
            "r2": r2
        }

        return self.metricas

    def prever(self, valores):
        """Realiza previsões"""
        valores = np.array(valores).reshape(-1, 1)
        return self.model.predict(valores)


class LogLogRegressionModel:
    def __init__(self, df, col_x, col_y):
        self.df = df
        self.col_x = col_x
        self.col_y = col_y
        self.model = LinearRegression()
        self.metricas = None

    def _validar(self):
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame vazio.")

        for col in [self.col_x, self.col_y]:
            if col not in self.df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada.")

        # log não aceita valores <= 0
        if (self.df[self.col_x] <= 0).any() or (self.df[self.col_y] <= 0).any():
            raise ValueError("Dados devem ser positivos para modelo log-log.")

    def treinar(self):
        """Treina regressão log-log"""
        self._validar()

        log_x = np.log10(self.df[self.col_x].values).reshape(-1, 1)
        log_y = np.log10(self.df[self.col_y].values)

        self.model.fit(log_x, log_y)

        y_pred = self.model.predict(log_x)

        # cálculo manual do R²
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        coef = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)

        self.metricas = {
            "coeficiente": coef,
            "intercepto": intercept,
            "r2": float(r2)
        }

        return self.metricas

    def prever(self, valores):
        """Previsão convertendo de volta do log"""
        valores = np.array(valores)

        if (valores <= 0).any():
            raise ValueError("Valores devem ser positivos para previsão log-log.")

        log_valores = np.log10(valores).reshape(-1, 1)
        log_pred = self.model.predict(log_valores)

        return 10 ** log_pred