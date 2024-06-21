import pandas as pd
import numpy as np
import random
import copy

from scipy.stats import mode
from typing import Optional, Union, Callable

class MyBaggingClf():
    """ Класс для работы с моделью бэггинга в задаче бинарной классификации. """
    
    def __init__(self,
                 estimator = None,
                 n_estimators: int = 10,
                 max_samples: float = 1.0,
                 oob_score: Optional[str] = None,
                 random_state: int = 42,
                ) -> None:
        """ Конструктор для экземпляра класса MyBaggingClf. 

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
        self.oob_score = oob_score
        self.oob_score_ = None
        
        self.estimators = []
     
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ Выполняет построение модели бэггинга.

        Args:
            X: Датасет с признаками, используемый для обучения.
            y: Истинные значения целевой переменной, используемые для обучения. 

        Returns:
            None

        """
        random.seed(self.random_state)

        # бустрэп 
        rows_num_list = X.index.to_list()
        rows_smpl_cnt = round(self.max_samples * X.shape[0]) 
        bootstrap_idxs = [random.choices(rows_num_list, k=rows_smpl_cnt) for _ in range(self.n_estimators)]
        
        oob_preds_bootstrap = {}
        
        # обучение базовых моделей 
        for i in range(self.n_estimators):
            rows_idxs = bootstrap_idxs[i]
            X_sample = X.loc[rows_idxs, :]
            y_sample = y.loc[rows_idxs]
            
            X_oob = X.drop(index=rows_idxs)
            
            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X_sample, y_sample)
            
            self.estimators.append(estimator)
            
            estimator_preds_oob = estimator.predict_proba(X_oob)
            for idx, pred in zip(X_oob.index, estimator_preds_oob):
                oob_preds_bootstrap[idx] = oob_preds_bootstrap.setdefault(idx, []) + [pred]

        # вычисление OOB-оценки
        preds_oob = pd.Series({k: np.mean(v) for k, v in oob_preds_bootstrap.items()})
        y_oob = y.loc[preds_oob.index]
        self.oob_score_ = self._get_oob_score(y_oob, preds_oob, self.oob_score)
                
            
    def predict(self, X: pd.DataFrame, type: str) -> np.ndarray:
        """ Прогнозирует классы.

        Args:
            X: Датасет с признаками для прогнозирования. 
            type: Метод агрегации предсказаний моделей в ансамбле. 

        Returns:
            Предсказанные классы.

        """
        bagging_preds = []
        for estimator in self.estimators:
            preds = estimator.predict_proba(X)
            bagging_preds.append(preds)
        
        bagging_preds = np.array(bagging_preds)
        
        if type == 'mean':
            return (np.mean(bagging_preds, axis=0) > .5).astype(int)
        elif type == 'vote':
            bagging_preds = (bagging_preds > .5).astype(int)
            return mode(bagging_preds, axis=0)[0]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """ Прогнозирует вероятность класса 1. 
    
        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные вероятности. 

        """
        bagging_preds = []
        
        for estimator in self.estimators:
            preds = estimator.predict_proba(X)
            bagging_preds.append(preds)
            
        return np.mean(bagging_preds, axis=0)
    
    def _get_oob_score(self, y_true: np.array, y_pred_prob: pd.Series, metric: str) -> float:
        """ Вычисляет значение метрики качества классификации на OOB выборке.

        Args:
            y_true: Истинные значения классов целевой переменной.
            y_pred_prob: Предсказанные вероятности класса 1. 
            metric: Используемая метрика качества. 

        Returns:
            Значение метрики.

        """
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
            precision = self._get_oob_score(y_true, y_pred_prob, 'precision')
            recall = self._get_oob_score(y_true, y_pred_prob, 'recall')
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
            
        return f"{type(self).__name__}: {', '.join(params)}"

class treeNode():
    """ Структура данных для построения решающего дерева. """
    
    def __init__(
        self,
        feature: str = None,
        threshold: float = None,
        left = None,
        right = None,
        ig: float = None,
        value: float = None
    ) -> None:
        # decision node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        
        self.value = value
        
    def is_leaf_node(self) -> bool:
        """ Индикатор листа. 

        Returns:
            True если рассматриваемый узел является листом.

        """
        return self.value is not None

class MyTreeClf():
    """ Класс для работы с моделью решающего дерева в задаче бинарной классификации. """
    
    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: Optional[int] = None,
                 criterion: str = 'entropy'
                ) -> None:
        """ Конструктор экземпляра класса MyTreeClf. 

        Args:
            max_depth: Максимальная глубина дерева.
            min_samples_split: Минимальное количество наблюдений для дальнейшего ветвления. 
            max_leafs: Максимальное количество листьев в дереве. 
            bins: Количество интервалов для поиска разбиений через гистограммный метод.
            criterion: Критерий для поиска наилучшего разбиения при ветвлении.
            
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

        # критерий разбиения 
        self.criterion = criterion
        self.criterion_dict = {
            'entropy': self._get_entropy,
            'gini': self._get_gini
        }
        
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
        
        self.fi = dict.fromkeys(X.columns, 0)
        
        if self.bins:
            self.features_thresholds = self._get_hist_thresholds(X, self.bins)

        # начинаем построение дерева
        self.leafs_cnt = 1
        self.root = self._build_tree_recursive(X, y, curr_depth = 0)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """ Прогнозирует классы.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные классы.

        """
        y_pred_prob = self.predict_proba(X)
        y_pred = y_pred_prob.apply(lambda x: 1 if x > .5 else 0)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """ Прогнозирует вероятность класса 1. 
    
        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные вероятности. 

        """
        y_pred_prob = X.apply(lambda x: self._traverse_tree(x, self.root), axis=1)

        return y_pred_prob
    
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
        
        best_split = self._get_best_split(X, y, self.bins, self.criterion)
        
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

        f = self.criterion_dict[self.criterion]
        I = f(y)
        I_l = f(y.iloc[left_idxs])
        I_r = f(y.iloc[right_idxs])

        fi = N_p / N * (I - (N_l / N_p * I_l) - (N_r / N_p * I_r))

        return fi
    
    def _get_entropy(self, y: pd.Series) -> float:
        """ Вычисляет информационную энтропию. 

        Args:
            y: Истинные значения целевой переменной. 

        Returns:
            Информационная энтропия. 

        """
        probs = np.bincount(y) / len(y)
        s = - np.sum(probs * np.log2(probs + 1e-15))

        return s

    def _get_gini(self, y: pd.Series) -> float:
        """ Вычисляет коэффициент Джини.

        Args:
            y: Истинные значения целевой переменной. 

        Returns:
            Коэффициент Джини.

        """
        probs = np.bincount(y) / len(y)
        gini = 1 - np.sum(np.square(probs))

        return gini
        
    def _get_split_gain(self, left_split: pd.Series, right_split: pd.Series, criterion: str) -> float:
        """ Вычисляет прирост информации от рассматриваемого разбиения по заданному критерию.

        Args:
            left_split: Левая подвыборка истинных значений целевой переменной.
            right_split: Правая подвыборка истинных значений целевой переменной.
            criterion: Критерий для оценки прироста информации.

        Returns:
            Прирост информации.

        """
        
        n_left, n_right, N = len(left_split), len(right_split), len(left_split) + len(right_split)

        f = self.criterion_dict[criterion]
        
        current_value = f(np.concatenate((left_split, right_split)))
        left_split_value = f(left_split)
        right_split_value = f(right_split)

        gain = current_value - (n_left / N * left_split_value + n_right / N * right_split_value)

        return gain
    
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

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series, bins: Optional[int], criterion: str) -> dict:
        """ Выполняет поиск наилучшего разбиения.

        Args:
            X: Датасет с признаками.
            y: Истинные значения целевой переменной.
            bins: Количество интервалов для поиска разбиений через гистограммный метод.
            criterion: Критерий для поиска наилучшего разбиения. 

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
                    
                current_gain = self._get_split_gain(y.iloc[left_split_idxs], y.iloc[right_split_idxs], criterion) 
                    
                if current_gain > max_gain:
                    best_split['feature'] = feature
                    best_split['threshold'] = threshold
                    best_split['left_idxs'] = left_split_idxs
                    best_split['right_idxs'] = right_split_idxs
                    best_split['ig'] = current_gain
   
                    max_gain = current_gain
        
        return best_split

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
            
        return f"{type(self).__name__}: {', '.join(params)}"

class MyKNNClf():
    """ Класс для работы с моделью k ближайших соседей в задаче бинарной классификации. """
    
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
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Прогнозирует классы.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные классы.

        """
        score_positive_class, score_negative_class = self._get_class_scores(X, self.weight)
        
        # возвращается класс с наибольшим весом
        y_pred = (score_positive_class >= score_negative_class).astype(int)
   
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """ Прогнозирует вероятность класса 1.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные вероятности.

        """
        score_positive_class, score_negative_class = self._get_class_scores(X, self.weight)
        y_pred_prob = score_positive_class / (score_positive_class + score_negative_class)
        
        return y_pred_prob
    
    def _get_class_scores(self, X: pd.DataFrame, weight: str) -> tuple:
        """ Вычисляет взвешенные оценки классов среди ближайших соседей.

        Args:
            X: Датасет с признаками для прогнозирования.
            weight: Метод взвешивания ближайших соседей.

        Returns:
            Пара взвешенных оценок классов, e.g.:
            
            (Оценка вероятности класса 1, Оценка вероятности класса 0)

        """  
        n_rows_test = X.shape[0]
        k_nearest_y = []
        weights = []
        
        # для каждого прогнозируемого объекта :
        for test_idx in range(n_rows_test):
            test_obj = X.iloc[test_idx]
            distances = self._get_distances(self.metric, test_obj)
            
            # индексы ближайших соседей 
            k_nearest_idxs = np.argsort(distances)[:self.k]
            # классы ближайших соседей
            k_nearest_y.append(self._train_y.iloc[k_nearest_idxs].to_list())     
        
            # взвешивание
            if weight == 'rank':
                k_nearest_weights = 1 / np.arange(1, self.k + 1)
            elif weight == 'distance':
                k_nearest_weights = 1 / distances[k_nearest_idxs]
            else:
                k_nearest_weights = np.ones(self.k)
                
            weights.append(k_nearest_weights)

        # для каждого класса считаем взвешенный скор
        score_positive_class = np.sum(np.where(np.array(k_nearest_y) == 1, np.array(weights), 0), axis=1)
        score_negative_class = np.sum(np.where(np.array(k_nearest_y) == 0, np.array(weights), 0), axis=1)

        return score_positive_class, score_negative_class
    
    def _get_distances(self, metric: str, test_obj: pd.Series) -> np.ndarray:
        """ Вычисляет расстояния между прогнозируемым объектом и всеми объектами обучающей выборки.

        Args:
            test_obj: Объект для прогноза.
            metric: Используемая мера расстояния.

        Returns:
            Матрица расстояний.

        """
        train_objs = self._train_X
        
        if metric == 'euclidean':
            distances = np.sqrt(np.sum(np.square(test_obj - train_objs), axis=1))
        elif metric == 'manhattan':
            distances = np.sum(np.abs(test_obj - train_objs), axis=1)
        elif metric == 'chebyshev':
            distances = np.max(np.abs(test_obj - train_objs), axis=1)
        elif metric == 'cosine':
            distances = 1 - (test_obj * train_objs).sum(axis=1) / (np.sqrt(np.sum(np.square(test_obj))) * np.sqrt(np.sum(np.square(train_objs), axis=1)))
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
    
    def _get_loss(self, y_true: pd.Series, y_pred_prob: np.array, reg: Optional[str] = None) -> float:
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
    
    def _get_grad(self, X: pd.DataFrame, y_true: pd.Series, y_pred_prob: np.array, reg: Optional[str] = None, batch: Optional[list] = None) -> np.ndarray:
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
