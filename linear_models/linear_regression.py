import pandas as pd
import numpy as np
import random

from typing import Optional, Union, Callable

class MyLineReg():
    """ Класс для работы с моделью линейной регрессии. """
    
    def __init__(self, 
                 n_iter: int = 100, 
                 learning_rate: Union[Callable, float] = .1, 
                 metric: Optional[str] = None,
                 reg: Optional[str] = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 sgd_sample: Optional[float] = None,
                 random_state: int = 42
                ) -> None:
        """ Конструктор для экземпляра класса MyLineReg.

        Args:
            n_iter: Количество итераций градиентного спуска.
            learning_rate: Скорость обучения.
            metric: Используемая метрика качества модели.
            reg: Метод регуляризации модели.
            l1_coef: Коэффициент l1 регуляризации.
            l2_coef: Коэффициент l2 регуляризации.
            sgd_sample: Доля наблюдений используемая для обучения через стохастический градиентный спуск.
            random_state: Сид для воспроизводимости результатов.

        Returns:
            None

        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        
        assert metric in ('mae', 'mse', 'rmse', 'mape', 'r2', None), f'{metric} is not supported'
        self.metric = metric
        
        assert reg in ('l1', 'l2', 'elasticnet', None), f'{reg} is not supported'
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Optional[int] = None) -> None:
        """ Выполняет построение модели линейной регрессии. 

        Args:
            X: Датасет с признаками, используемый для обучения.
            y: Истинные значения целевой переменной, используемые для обучения.
            verbose: Контролирует вывод информации о процессе обучения.

        Returns:
            None

        """
        random.seed(self.random_state)
        self.metric_values = []
        
        n_rows, n_features = X.shape
        # стохастический градиентный спуск
        if self.sgd_sample is not None:
            if not isinstance(self.sgd_sample, int):
                sgd_sample = round(n_rows * self.sgd_sample)
            else:
                sgd_sample = self.sgd_sample
        
        # добавляем единичный стобец в матрицу признаков
        X = np.hstack((np.ones((n_rows, 1)), X))
        # инициализируем веса
        if self.weights is None:
            self.weights = np.ones(n_features + 1)

        
        base_score = np.mean(np.square((y - X @ self.weights)))
        for i in range(1, self.n_iter + 1):       
            loss = self._get_loss(X, y, self.reg)
            
            # вычисляем градиент
            if self.sgd_sample is not None:
                sample_rows_idx = random.sample(range(n_rows), sgd_sample)
                grad = self._get_grad(X, y, self.reg, batch=sample_rows_idx)
            else:
                grad = self._get_grad(X, y, self.reg)
                
            # обновляем веса
            self.weights -= self._get_lr(i) * grad
            
            # метрика
            if self.metric:
                metric_score = self._get_score(y, X @ self.weights, self.metric)
                self.metric_values.append(metric_score)
            # вывод информации о процессе обучения
            if verbose:
                if i == 1:
                    log = f'start | loss: {base_score}'
                    if self.metric:
                        log += f' | {self.metric}: {metric_score}'
                    print(log)
                elif i % verbose == 0:
                    log = f'{i} | loss: {loss}'
                    if self.metric:
                        log += f' | {self.metric}: {metric_score}'
                    print(log)
                else:
                    pass
            
    def get_coef(self) -> np.array:
        """ Коэффициенты обученной модели.

        Returns:
            Коэффициенты.

        """
        return self.weights[1:].copy()
    
    def predict(self, X: pd.DataFrame) -> np.array:
        """ Прогнозирует целевую переменную.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные значения.

        """
        
        n_rows = X.shape[0]
        X = np.hstack((np.ones((n_rows, 1)), X))
        y_pred = X @ self.weights
        
        return y_pred
    
    def get_best_score(self) -> float:
        """ Метрика качества обученной модели. 

        Returns:
            Значение метрики.

        """
        return self.metric_values[-1]

    def _get_score(self, y_true: np.array, y_pred: np.array, metric: str) -> float:
        """ Вычисляет значение метрики качества регрессии.

        Args:
            y_true: Истинные значения целевой переменной.
            y_pred: Предсказанные значения целевой переменной. 
            metric: Используемая метрика качества. 

        Returns:
            Значение метрики.

        """
        if metric == 'mae':
            score = np.mean(np.abs(y_true - y_pred))
        elif metric == 'mse':
            score = np.mean(np.square(y_true - y_pred))
        elif metric == 'rmse':
            score = np.sqrt(np.mean(np.square(y_true - y_pred)))
        elif metric == 'mape':
            score = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
        elif metric == 'r2':
            score = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
        else:
            pass
        
        return score
    
    def _get_grad(self, X: pd.DataFrame, y: pd.Series, reg: Optional[str], batch: Optional[list] = None) -> np.array:
        """ Вычисляет градиент функции потерь.

        Args:
            X: Датасет с признаками, используемый для обучения 
            y: Истинные значения целевой переменной, используемые для обучения.
            reg: Метод регуляризации.
            batch: Индексы объектов в батче.

        Returns:
            Градиент. 

        """
        
        if batch is not None:
            X = X[batch]
            y = y.iloc[batch]
            
        n_rows = X.shape[0]
        y_pred = X @ self.weights
        grad = 2 / n_rows * (y_pred - y) @ X
        
        if reg == 'l1':
            reg_term = self.l1_coef * np.sign(self.weights)
        elif reg == 'l2':
            reg_term = self.l2_coef * 2 * self.weights
        elif reg == 'elasticnet':
            reg_term = self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        else:
            reg_term = 0
            
        return grad + reg_term
    
    def _get_loss(self, X: pd.DataFrame, y: pd.Series, reg: Optional[str]) -> float:
        """ Вычисляет значение функции потерь. 

        Args:
            X: Датасет с признаками, используемый для обучения 
            y: Истинные значения целевой переменной, используемые для обучения.
            reg: Метод регуляризации.
        Returns:
            Значение функции потерь 

        """
        y_pred = X @ self.weights
        loss = np.mean(np.square((y - y_pred)))
        
        if reg == 'l1':
            reg_term = self.l1_coef * np.sum(np.abs(self.weights))
        elif reg == 'l2':
            reg_term = self.l2_coef * np.sum(np.square(self.weights))
        elif reg == 'elasticnet':
            reg_term = self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(np.square(self.weights))
        else:
            reg_term = 0
            
        return loss + reg_term

    def _get_lr(self, i: int) -> float:
        """ Вычисляет значение текущей скорости обучения.

        Args:
            i: Текущий шаг обучения.

        Returns:
            Текущая скорость обучения.

        """   
        if isinstance(self.learning_rate, (int, float)):
            return self.learning_rate
        else:
            return self.learning_rate(i)

    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f"{type(self).__name__} class: {', '.join(params)}"
