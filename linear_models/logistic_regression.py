import pandas as pd
import numpy as np
import random

from typing import Union, Optional, Callable 

class MyLogReg():
    """ Класс для работы с моделью логистической регрессии в задаче бинарной классификации. """
    
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate: Union[Callable, float] = .1,
                 metric: str = None,
                 reg: str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 sgd_sample: Optional[Union[int, float]] = None,
                 random_state: int = 42
                ) -> None:
        """ Конструктор для экземпляра класса MyLogReg. 

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

        assert metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', None), f'{metric} metric is not supported'
        self.metric = metric

        assert reg in ('l1', 'l2', 'elasticnet', None), f'{reg} regularization is not supported'
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        
    def get_coef(self) -> np.ndarray:
        """ Коэффициенты обученной модели.

        Returns:
            Коэффициенты.

        """
        return self.weights[1:].copy()
     
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Optional[int] = None) -> None:
        """ Выполняет построение модели логистической регрессии. 

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

        
        for i in range(1, self.n_iter + 1):
            # лосс
            y_pred_prob = self._sigmoid(X @ self.weights)
            logloss = self._get_loss(y, y_pred_prob, self.reg)
            
            # вычисляем градиент
            if self.sgd_sample is not None:
                batch_rows_idx = random.sample(range(n_rows), sgd_sample)
                grad = self._get_grad(X, y, y_pred_prob, self.reg, batch=batch_rows_idx)
            else:
                grad = self._get_grad(X, y, y_pred_prob, self.reg)
            
            # обновляем веса
            self.weights -= self._get_lr(i) * grad
            
            # метрика 
            if self.metric:
                metric = self._get_score(X, y, self.metric)
                self.metric_history.append(metric)
            # вывод информации о процессе обучения 
            if verbose and i % verbose == 0:
                log = f'{i} | loss: {logloss}'
                if self.metric:
                    log += f' | {self.metric}: {metric}'
                print(log)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Прогнозирует классы.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные классы.

        """
        y_pred_prob = self.predict_proba(X)
        y_pred = (y_pred_prob > .5).astype(int)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """ Прогнозирует вероятность класса 1. 
    
        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные вероятности. 

        """
        n_rows = X.shape[0]
        X = np.hstack((np.ones((n_rows, 1)), X))
        
        logits = X @ self.weights 
        y_pred_prob = self._sigmoid(logits)
        
        return y_pred_prob
    
    def get_best_score(self) -> float:
        """ Метрика качества обученной модели. 

        Returns:
            Значение метрики.

        """
        return self.metric_history[-1]

    def _sigmoid(self, z: float) -> float:
        """ Приводит логиты к вероятностям.

        Args:
            z: Логит.

        Returns:
            Вероятность.

        """
        return 1.0 / (1.0 + np.exp(-z))
    
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
    
    def _get_loss(self, y_true: pd.Series, y_pred_prob: np.ndarray, reg: Optional[str] = None) -> float:
        """ Вычисляет значение функции потерь. 

        Args:
            y_pred_prob: Предсказанные вероятности. 
            y_true: Истинные значения целевой переменной, используемые для обучения.
            reg: Метод регуляризации.
        Returns:
            Значение функции потерь 

        """
        
        loss = -np.mean(y_true * np.log(y_pred_prob + 1e-15) + (1 - y_true) * np.log(1 - y_pred_prob + 1e-15))
        
        if reg == 'l1':
            reg_term = self.l1_coef * np.sum(np.abs(self.weights))
        elif reg == 'l2':
            reg_term = self.l2_coef * np.sum(np.square(self.weights))
        elif reg == 'elasticnet':
            reg_term = self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(np.square(self.weights))
        else:
            reg_term = 0
        
        return loss + reg_term
    
    def _get_grad(self, X: pd.DataFrame, y_true: pd.Series, y_pred_prob: np.ndarray, reg: Optional[str] = None, batch: Optional[list] = None) -> np.ndarray:
        """ Вычисляет градиент функции потерь.

        Args:
            X: Датасет с признаками, используемый для обучения 
            y_true: Истинные значения целевой переменной, используемые для обучения.
            y_pred_prob: Предсказанные вероятности.
            reg: Метод регуляризации.
            batch: Индексы объектов в батче.

        Returns:
            Градиент. 

        """
        
        if batch is not None:
            X = X[batch]
            y_true = y_true.iloc[batch]
            y_pred_prob = y_pred_prob[batch]
            
        n_rows = X.shape[0]
        grad = (y_pred_prob - y_true) @ X / n_rows
        
        if reg == 'l1':
            reg_term = self.l1_coef * np.sign(self.weights)
        elif reg == 'l2':
            reg_term = self.l2_coef * 2 * self.weights 
        elif reg == 'elasticnet':
            reg_term = self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        else:
            reg_term = 0
        
        return grad + reg_term
    
    def _get_score(self, X: pd.DataFrame, y_true: pd.Series, metric: Optional[str]) -> float:
        """ Вычисляет значение метрики качества бинарной классификации.

        Args:
            X: Датасет с признаками. 
            y_true: Истинные значения целевой переменной. 
            metric: Используемая метрика качества. 

        Returns:
            Значение метрики.

        """
        
        y_pred_prob = self._sigmoid(X @ self.weights)
        y_pred = (y_pred_prob > .5)
        
        if metric == 'accuracy':
            total = y_true.shape[0]
            tp = np.sum((y_pred == 1) & (y_pred == y_true))
            tn = np.sum((y_pred == 0) & (y_pred == y_true)) 
            score = (tp + tn) / total
        elif metric == 'precision':
            tp = np.sum((y_pred == 1) & (y_pred == y_true))
            fp = np.sum((y_pred == 1) & (y_pred != y_true))
            score = tp / (tp + fp)
        elif metric == 'recall':
            tp = np.sum((y_pred == 1) & (y_pred == y_true))
            fn = np.sum((y_pred == 0) & (y_pred != y_true)) 
            score = tp / (tp + fn)
        elif metric == 'f1':
            precision = self._get_score(X, y_true, 'precision')
            recall = self._get_score(X, y_true, 'recall')
            score = 2 * precision * recall / (precision + recall)
        elif metric == 'roc_auc':
            y_pred_prob = np.round(y_pred_prob, 10)
            prob_target_pairs = [*zip(y_pred_prob, y_true)]
            sorted_pairs = sorted(prob_target_pairs, key=lambda x: x[0], reverse=True)
            
            n_positives = np.sum((y_true == 1))
            n_negatives = np.sum((y_true == 0))
            curr_score = 0
            for pair in sorted_pairs:
                if pair[1] == 0:
                    curr_prob = pair[0]
                    higher_prob_positives = [pair for pair in sorted_pairs if pair[0] > curr_prob and pair[1] == 1]
                    same_prob_positives = [pair for pair in sorted_pairs if pair[0] == curr_prob and pair[1] == 1]
                    
                    curr_score += len(higher_prob_positives) + len(same_prob_positives) / 2         
            score = curr_score / (n_positives * n_negatives)
        
        return score

    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f"{type(self).__name__} class: {', '.join(params)}"
