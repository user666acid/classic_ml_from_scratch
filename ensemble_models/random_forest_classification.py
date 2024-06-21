import pandas as pd
import numpy as np
import random

from typing import Optional
from scipy.stats import mode

class MyForestClf():
    """ Класс для работы с моделью случайного леса в задаче бинарной классификации. """
    
    def __init__(self,
                 n_estimators: int = 10,
                 max_features: float = .5,
                 max_samples: float = .5,
                 criterion: str = 'entropy',
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: Optional[int] = 16,
                 oob_score: Optional[str] = None,
                 random_state: int = 42
                ) -> None:
        """ Конструктор для экземпляра класса MyForestClf.

        Args:
            n_estimators: Количество деревьев.
            max_features: Для признаков используемая для обучения отдельного дерева.
            max_samples: Доля наблюдений используемая для обучения отдельного дерева.
            criterion: Критерий для поиска наилучшего разбиения при ветвлении. 
            max_depth: Максимальная глубина дерева.
            min_samples_split: Минимальное количество наблюдений для дальнейшего ветвления.
            max_leafs: Максимальное количество листьев в дереве.
            bins: Количество интервалов для поиска разбиений через гистограммный метод.
            oob_score: Метрика для OOB-оценки качества дерева.
            random_state: Сид для воспроизводимости результатов.

        Returns:
            None

        """
        # параметры леса
        self.random_state = random_state
        self.trees = []
        self.n_estimators = n_estimators
        self.oob_score = oob_score

        self.max_features = max_features
        self.max_samples = max_samples

        # параметры дерева 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.criterion = criterion
        
        # использование гистограмм для сплитов    
        self.bins = bins
        self.features_thresholds = None

        # важность признаков 
        self.N = 0
        self.fi = {} 
        
        self.leafs_cnt = 0
        self.oob_score_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ Выполняет построение модели случайного леса.

        Args:
            X: Датасет с признаками, используемый для обучения.
            y: Истинные значения целевой переменной, используемые для обучения. 

        Returns:
            None

        """
        random.seed(self.random_state)
        
        self.N = X.shape[0]
        self.fi = dict.fromkeys(X.columns, 0)
        
        init_cols = X.columns.to_list()
        n_cols_sample = round(self.max_features * len(init_cols))
        n_rows_sample = round(self.max_samples * self.N)
        
        oob_preds_forest = {}
        
        for _ in range(self.n_estimators):
            cols_idxs = random.sample(init_cols, n_cols_sample)
            rows_idxs = random.sample(range(self.N), n_rows_sample)
            
            # отбираем подвыборку
            X_sample, X_oob = X.loc[rows_idxs, cols_idxs], X.loc[:, cols_idxs].drop(index=rows_idxs)
            y_sample = y.loc[rows_idxs]   
            
            # создаем, обучаем и сохраняем дерево 
            tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X_sample, y_sample, self.N)
            self.trees.append(tree)

            # подсчёт листьев, важность признаков
            self.leafs_cnt += tree.leafs_cnt
            for col in cols_idxs:
                self.fi[col] += tree.fi[col]
                
            # получаем предикты для подмножества rows_idxs_oob
            tree_preds_oob = tree.predict_proba(X_oob)
            # сохраняем предикты и индексы 
            for idx, pred in zip(X_oob.index, tree_preds_oob):
                oob_preds_forest[idx] = oob_preds_forest.setdefault(idx, []) + [pred]
                
        # усредняем oob вероятности
        preds_oob = pd.Series({k: np.mean(v) for k, v in oob_preds_forest.items()})  
        y_oob = y.loc[preds_oob.index]    
        self.oob_score_ = self._get_oob_score(y_oob, preds_oob, self.oob_score)
    
    def predict(self, X: pd.DataFrame, type: str) -> np.ndarray:
        """ Прогнозирует классы.

        Args:
            X: Датасет с признаками для прогнозирования. 
            type: Метод агрегации предсказаний деревьев в лесу. 

        Returns:
            Предсказанные классы.

        """
        tree_preds = []
        for tree in self.trees:
            preds = tree.predict_proba(X)
            tree_preds.append(preds)
        
        tree_preds = np.array(tree_preds)
        
        if type == 'mean':
            return (np.mean(tree_preds, axis=0) > .5).astype(int)
        elif type == 'vote':
            tree_preds = (tree_preds > .5).astype(int)
            return mode(tree_preds, axis=0)[0]
        
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """ Прогнозирует вероятность класса 1. 
    
        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные вероятности. 

        """
        tree_preds = []
        for tree in self.trees:
            preds = tree.predict_proba(X)
            tree_preds.append(preds)
            
        return np.mean(tree_preds, axis=0)
                                              
    def _get_oob_score(self, y_true: np.ndarray, y_pred_prob: pd.Series, metric: str) -> float:
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
                 bins: Optional[int] = 20,
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

    def fit(self, X: pd.DataFrame, y: pd.Series, init_N: int) -> None:
        """ Инициирует процедуру построения решающего дерева. 

        Args:
            X: Датасет с признаками.
            y: Истинные значения целевой переменной.
            init_N: Количество наблюдений в исходном датасете, используется при расчёте важности признаков.

        Returns:
            None

        """
        self.N = init_N
        
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
