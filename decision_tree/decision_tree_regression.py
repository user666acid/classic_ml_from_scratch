import pandas as pd
import numpy as np

from typing import Optional

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
    """ Класс для работы с моделью решающего дерева для регрессии. """
    
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
