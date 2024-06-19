import pandas as pd
import numpy as np
import random
import copy

from typing import Optional, Callable, Union

class MyBaggingReg():
    """ Класс для работы с моделью бэггинга в задаче регрессии. """
    
    def __init__(self,
                 estimator = None,
                 n_estimators: int = 10,
                 max_samples: float = 1.0,
                 oob_score: Optional[str] = None,
                 random_state: int = 42        
                ) -> None:
        """ Конструктор для экземпляра класса MyBaggingReg.

        Args:
            estimator: Базовая модель.
            n_estimators: Количество моделей в ансамбле.
            max_samples: Доля наблюдений для бутстрэпа.
            oob_score: Метрика для OOB-оценки качества ансамбля.
            random_state: Сид для воспроизводимости результатов.

        Returns:
            None

        """
        self.estimator = estimator 
        self.n_estimators = n_estimators 
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.estimators = []
        self.oob_score = oob_score
        self.oob_score_ = 0
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ Выполняет построение модели бэггинга.

        Args:
            X: Датасет с признаками, используемый для обучения.
            y: Истинные значения целевой переменной, используемые для обучения. 

        Returns:
            None

        """
        random.seed(self.random_state)
         
        rows_num_list = X.index.to_list()
        rows_smpl_cnt = round(self.max_samples * X.shape[0])
    
        bootstrap_idxs = [random.choices(rows_num_list, k=rows_smpl_cnt) for _ in range(self.n_estimators)]
        oob_preds_bagging = {}
            
        for i in range(self.n_estimators):
            rows_idxs = bootstrap_idxs[i]
            # отбираем подвыборку для обучения и для oob score
            X_sample, X_oob = X.loc[rows_idxs, :], X.drop(index=rows_idxs)
            y_sample = y.loc[rows_idxs]
            
            # создаем и обучаем модель
            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X_sample, y_sample)
       
            # сохраняем
            self.estimators.append(estimator)
           
            estimator_oob_preds = estimator.predict(X_oob)
            for idx, pred in zip(estimator_oob_preds.index, estimator_oob_preds):
                oob_preds_bagging[idx] = oob_preds_bagging.setdefault(idx, []) + [pred]
                
        oob_preds = pd.Series({k: np.mean(v) for k, v in oob_preds_bagging.items()})
        y_oob = y.loc[oob_preds.index]
        self.oob_score_ = self._get_oob_score(y_oob, oob_preds, self.oob_score)
    
    def predict(self, X: pd.DataFrame) -> np.array:
        """ Прогнозирует целевую переменную.

        Args:
            X: Датасет с признаками для прогнозирования. 

        Returns:
            Предсказанные значения.

        """
        
        bagging_preds = []
        for estimator in self.estimators:
            preds = estimator.predict(X)
            bagging_preds.append(preds)
        
        return np.mean(bagging_preds, axis=0)
    
    def _get_oob_score(self, y_true: np.array, y_pred: pd.Series, metric: str) -> float:
        """ Вычисляет значение метрики качества регрессии на OOB выборке.

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
            y_mean = np.mean(y_true)
            score = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - y_mean))
        else:
            pass
        
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
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """ Прогнозирует целевую переменную.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные значения.

        """
        index = X.index
        n_rows = X.shape[0]
        X = np.hstack((np.ones((n_rows, 1)), X))
        y_pred = pd.Series(X @ self.weights, index=index)
        
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
    
    def _get_grad(self, X: pd.DataFrame, y: pd.Series, reg: Optional[str] = None, batch: Optional[list] = None) -> np.array:
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
    
    def _get_loss(self, X: pd.DataFrame, y: pd.Series, reg: Optional[str] = None) -> float:
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

class MyKNNReg():
    """ Класс для работы с моделью k ближайших соседей в задаче регрессии. """
    
    def __init__(self, 
                 k: int = 3,
                 metric: str = 'euclidean',
                 weight: str = 'uniform'
                ) -> None:
        """ Конструктор для экземпляра класса MyKNNClf. 

        Args:
            k: Количество соседей.
            metric: Используемая мера расстояния между точками.
            weight: Метод взвешивания ближайших соседей. 

        Returns:
            None

        """
        self.k = k
        self.train_size = None
        self._train_X = None
        self._train_y = None
        self.metric = metric
        self.weight = weight
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ Запоминает обучающую выборку. 

        Args:
            X: Датасет с признаками, используемый для обучения.
            y: Истинные значения целевой переменной, используемые для обучения. 

        Returns:
            None

        """
        self._train_X = X.reset_index(drop=True).copy()
        self._train_y = y.reset_index(drop=True).copy()
        self.train_size = X.shape


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """ Прогнозирует целевую переменную.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные значения целевой переменной.

        """
        
        y_pred = pd.Series(None, index=X.index)
        
        for i in range(len(X)):
            distances = self._get_distances(X.iloc[i].values, self.metric)

            k_nearest_idxs = distances.sort_values()[:self.k].index
            k_nearest_y = self._train_y.loc[k_nearest_idxs]
        
            if self.weight == 'uniform':
                weights = np.ones(self.k)
                
            elif self.weight == 'rank':
                weights = 1 / np.arange(1, self.k+1)

            elif self.weight == 'distance':
                k_nearest_distances = np.sort(distances)[:self.k]
                weights = 1 / k_nearest_distances     
            
            weights = weights / np.sum(weights)
            weighted_pred = np.sum(k_nearest_y * weights)
            
            y_pred.iloc[i] = weighted_pred
        
        return y_pred
 
    def _get_distances(self, test_obj: pd.Series, metric: str) -> np.ndarray:
        """ Вычисляет расстояния между прогнозируемым объектом и всеми объектами обучающей выборки.

        Args:
            test_obj: Объект для прогноза.
            metric: Используемая мера расстояния.

        Returns:
            Матрица расстояний.

        """
        train_objs = self._train_X
        
        if metric == 'euclidean':
            distances = np.sqrt(np.sum(np.square(train_objs - test_obj), axis=1))
        elif metric == 'manhattan':
            distances = np.sum(np.abs(train_objs - test_obj), axis=1)
        elif metric == 'chebyshev':
            distances = np.max(np.abs(train_objs - test_obj), axis=1)
        elif metric == 'cosine':
            distances = 1 - (train_objs - test_obj).sum(axis=1) \
                / (np.sqrt(np.sum(np.square(test_obj))) * np.sqrt(np.sum(np.square(train_objs), axis=1)))
        else:
            raise ValueError(f'Metric {metric} is not supported')
            
        return distances
    
    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f"{type(self).__name__} class: {', '.join(params)}"

class treeNode():
    """ Структура данных для построения решающего дерева. """
    
    def __init__(self,
                 feature: str = None,
                 threshold: float = None,
                 left = None,
                 right = None,
                 ig: float = None,
                 value: float = None
                ) -> None:
        """ Конструктор для экземпляра класса treeNode. 

        Args:
            feature: Признак по которому проводится разбиение.
            threshold: Пороговое значение для разбиения.
            left: Левая ветвь.
            right: Правая ветвь.
            ig: Информационный прирост от разбиения.
            value: Значение в листе.

        Returns:
            None

        """
        # decision node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.ig = ig
        
        # leaf node
        self.value = value
        
    def is_leaf_node(self) -> bool:
        """ Индикатор листа. 

        Returns:
            True если рассматриваемый узел является листом.

        """
        return self.value is not None

class MyTreeReg():
    """ Класс для работы с моделью решающего дерева в задаче регрессии. """
    
    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: Optional[int] = None
                ) -> None:
        """ Конструктор экземпляра класса MyTreeReg. 

        Args:
            max_depth: Максимальная глубина дерева.
            min_samples_split: Минимальное количество наблюдений для дальнейшего ветвления. 
            max_leafs: Максимальное количество листьев в дереве. 
            bins: Количество интервалов для поиска разбиений через гистограммный метод.
            
        Returns:
            None

        """
        # ограничения 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs if max_leafs > 1 else 2
        
        # использование гистограмм для сплитов
        self.bins = bins
        self.features_thresholds = None
    
        # важность признаков
        self.N = 0
        self.fi = {}

        # построение дерева
        self.leafs_cnt = 0
        self.leafs_sum = 0
        
        self.root = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ Инициирует процедуру построения решающего дерева. 

        Args:
            X: Датасет с признаками.
            y: Истинные значения целевой переменной.

        Returns:
            None

        """
        self.N = X.shape[0]

        # важность признаков 
        for col in X.columns:
            self.fi[col] = 0

        # гистограммный метод
        if self.bins:
            self.features_thresholds = self._get_hist_thresholds(X, self.bins)
        
        self.leafs_cnt = 1
        self.root = self._build_tree_recursive(X, y, curr_depth = 0)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """ Прогнозирует целевую переменную.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные значения целевой переменной.

        """
        y_pred = X.apply(lambda x: self._traverse_tree(x, self.root), axis=1)
        
        return y_pred
        
    def _traverse_tree(self, row: pd.Series, node: treeNode) -> float:
        """ Проход по построенному дереву. 

        Args:
            row: Наблюдение, для которого выполняется проход по дереву.
            node: Корень построенного дерева.

        Returns:
            Значение листа.

        """
        if node.is_leaf_node():
            return node.value
        
        if row[node.feature] <= node.threshold:
            return self._traverse_tree(row, node.left)
        
        return self._traverse_tree(row, node.right)

    def _build_tree_recursive(self, X: pd.DataFrame, y: pd.Series, curr_depth: int = 0) -> treeNode:  
        """ Выполняет рекурсивное построение дерева решений.

        Args:
            X: Датасет с признаками.
            y: Истинные значения целевой переменной.
            curr_depth: Текущая глубина дерева.

        Returns:
            Корень построенного дерева.

        """

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            self.leafs_cnt >= self.max_leafs
            or n_labels <= 1
        ):
            return self._create_leaf_node(y)
        
        best_split = self._get_best_split(X, y, self.bins)

        if not best_split:
            return self._create_leaf_node(y)
            
        decision_node = treeNode(best_split['feature'], best_split['threshold'], ig=best_split['ig'])

        if curr_depth < self.max_depth and n_samples >= self.min_samples_split:
            self.leafs_cnt += 1
            left_idxs, right_idxs = best_split['left_idxs'], best_split['right_idxs']
            
            # считаем FI
            fi = self._get_fi(y, left_idxs, right_idxs)
            self.fi[best_split['feature']] += fi
            
            decision_node.left = self._build_tree_recursive(X.iloc[left_idxs, :], y.iloc[left_idxs], curr_depth + 1)
            decision_node.right = self._build_tree_recursive(X.iloc[right_idxs, :], y.iloc[right_idxs], curr_depth + 1)
        else:
            return self._create_leaf_node(y)

        return decision_node
    
    def _get_fi(self, y: pd.Series, left_idxs: np.ndarray, right_idxs: np.ndarray) -> float:
        """ Вычисляет важность для рассматриваемого признака. 

        Args:
            y: Истинные значения целевой переменной.
            left_idxs: Индексы левой подвыборки. 
            right_idxs: Индексы правой подвыборки.

        Returns:
            Важность признака.

        """
        N = self.N
        N_p = y.shape[0]
        N_l, N_r = len(left_idxs), len(right_idxs)

        I = self._get_mse(y)
        I_l = self._get_mse(y.iloc[left_idxs])
        I_r = self._get_mse(y.iloc[right_idxs])

        fi = N_p / N * (I - (N_l / N_p * I_l) - (N_r / N_p * I_r))

        return fi
    
    def _get_mse(self, y: pd.Series) -> float:
        """ Вычисляет среднеквадратичную ошибку, в качестве прогноза используется среднее значение.

        Args:
            y: Истинные значения целевой переменной.

        Returns:
            Среднеквадратичная ошибка.

        """
        y_mean = np.mean(y)
        mse = np.mean(np.square(y - y_mean))
    
        return mse

    def _get_split_gain(self, left_split: pd.Series, right_split: pd.Series) -> float:
        """ Вычисляет прирост информации от рассматриваемого разбиения.

        Args:
            left_split: Левая подвыборка истинных значений целевой переменной.
            right_split: Правая подвыборка истинных значений целевой переменной.

        Returns:
            Прирост информации.

        """
        n_left, n_right, N = len(left_split), len(right_split), len(left_split) + len(right_split)

        current_mse = self._get_mse(np.concatenate((right_split, left_split)))
        left_split_mse = self._get_mse(left_split)
        right_split_mse = self._get_mse(right_split)

        gain = current_mse - (n_left / N * left_split_mse + n_right / N * right_split_mse)

        return gain

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series, bins) -> dict:
        """ Выполняет поиск наилучшего разбиения.

        Args:
            X: Датасет с признаками.
            y: Истинные значения целевой переменной.
            bins: Количество интервалов для поиска разбиений через гистограммный метод.

        Returns:
            Словарь содержащий информацию о наилучшем разбиении, e.g.:

            {'feature': Название признака,
             'threshold': Пороговое значение,
             'left_idxs': Индексы левой подвыборки,
             'right_idxs': Индексы правой подвыборки,
             'ig': Прирост информации}

        """
        best_split = {}
        max_gain = -np.inf

        for feature, values in X.items():
            if bins:
                thresholds = self.features_thresholds[feature]
            else:
                unique_values = sorted(values.unique())
                n_thresholds = len(unique_values) - 1
                thresholds = [np.mean(unique_values[i: i + 2]) for i in range(n_thresholds)]

            for threshold in thresholds:
                left_split_idxs = np.argwhere(X[feature].values <= threshold).flatten()
                right_split_idxs = np.argwhere(X[feature].values > threshold).flatten()

                current_gain = self._get_split_gain(y.iloc[left_split_idxs], y.iloc[right_split_idxs]) 

                if current_gain > max_gain:
                    best_split['feature'] = feature
                    best_split['threshold'] = threshold
                    best_split['left_idxs'] = left_split_idxs
                    best_split['right_idxs'] = right_split_idxs
                    best_split['ig'] = current_gain

                    max_gain = current_gain

        return best_split
    
    def _get_hist_thresholds(self, X: pd.DataFrame, bins: int) -> dict:
        """ Вычисляет пороговые значения при гистограммном методе.

        Args:
            X: Датасет с признаками.
            bins: Количество интервалов для поиска разбиений.

        Returns:
            Словарь содержащий пороговые значения для всех признаков, e.g.:

            {'col_0': [0.3, 0.6, 0.9],
             'col_1': [0.15, 0.45, 0.75]}

        """
        features_thresholds = {}
        
        for feature, values in X.items():
            unique_values = sorted(values.unique())
            n_thresholds = len(unique_values) - 1
            
            if n_thresholds <= bins - 1:
                thresholds = [np.mean(unique_values[i: i + 2]) for i in range(n_thresholds)]
            else:
                thresholds = np.histogram(values, bins)[1][1: -1]
            
            features_thresholds[feature] = thresholds
            
        return features_thresholds

    def _create_leaf_node(self, y: pd.Series) -> treeNode:
        """ Объявляет рассматриваемый узел листом. 

        Args:
            y: Истинные значения целевой переменной.

        Returns:
            Лист дерева.

        """     
        leaf_value = np.mean(y)
        self.leafs_sum += leaf_value
        
        leaf_node = treeNode(value = leaf_value)
        
        return leaf_node
   
    def print_tree(self, tree=None, indent=' ') -> str:
        """ Выводит структуру дерева. 

        Args:
            tree: Корень дерева. 
            indent: Отступ при визуализации.

        Returns:
            Текстовое отображение структуры дерева.

        """
        if not tree:
            tree = self.root
            
        if tree.value is not None:
            print(tree.value)
        
        else:
            print(f'{tree.feature} > {tree.threshold} ? {tree.ig}')
            print(f'left: {indent}', end='')
            self.print_tree(tree.left, indent + indent)
            print(f'right: {indent}', end='')
            self.print_tree(tree.right, indent + indent)
    
    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f"{type(self).__name__} class: {', '.join(params)}"
