import pandas as pd
import numpy as np
import random

from typing import Optional

class MyForestReg():
    """ Класс для работы с моделью случайного леса для регрессии. """
    
    def __init__(self,
                 n_estimators: int = 10,
                 max_features: float = .5,
                 max_samples: float = .5,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 oob_score: Optional[str] = None,
                 bins: Optional[int] = 16,
                 random_state: int = 42
                ) -> None:
        """ Конструктор для экземпляра класса MyForestReg.

        Args:
            n_estimators: Количество деревьев.
            max_features: Для признаков используемая для обучения отдельного дерева.
            max_samples: Доля наблюдений используемая для обучения отдельного дерева.
            max_depth: Максимальная глубина дерева.
            min_samples_split: Минимальное количество наблюдений для дальнейшего ветвления.
            max_leafs: Максимальное количество листьев в дереве.
            oob_score: Метрика для OOB-оценки качества дерева.
            bins: Количество интервалов для поиска разбиений через гистограммный метод.
            random_state: Сид для воспроизводимости результатов.

        Returns:
            None

        """
        # параметры леса
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.trees = []
        self.oob_score = oob_score

        self.max_features = max_features
        self.max_samples = max_samples
     
        # ограничения 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        
        # использование гистограмм для сплоитов    
        self.bins = bins
        self.features_thresholds = None
        
        # важность признаков
        self.N = 0
        self.fi = {}
      
        self.leafs_cnt = 0
        self.oob_score_ = 0
    
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
        fself.fi = dict.fromkeys(X.columns, 0)
        
        init_cols = X.columns.to_list()
        init_rows_cnt = self.N
        cols_smpl_cnt = round(self.max_features * len(init_cols))
        rows_smpl_cnt = round(self.max_samples * init_rows_cnt)
        
        oob_preds_forest = {}
            
        for tree in range(self.n_estimators):
            cols_idxs = random.sample(init_cols, cols_smpl_cnt)
            rows_idxs = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            rows_idxs_oob = list(set(range(init_rows_cnt)) - set(rows_idxs))
            # отбираем подвыборку для обучения и для oob score
            X_sample, X_oob = X.loc[rows_idxs, cols_idxs], X.loc[rows_idxs_oob, cols_idxs]
            y_sample = y.loc[rows_idxs]
            # создаем и обучаем дерево 
            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X_sample, y_sample, self.N)
            # сохраняем
            self.trees.append(tree)
            
            # подсчёт листьев, важность признаков
            self.leafs_cnt += tree.leafs_cnt
            for col in cols_idxs:
                self.fi[col] += tree.fi[col]
                    
            # получаем предикты для подмножества rows_idxs_oob
            tree_oob_preds = tree.predict(X_oob)
            # сохраняем предикты и индексы 
            for idx, pred in zip(tree_oob_preds.index, tree_oob_preds):
                oob_preds_forest[idx] = oob_preds_forest.setdefault(idx, []) + [pred]
        
        # усредняем oob предикты
        oob_preds = pd.Series({k: np.mean(v) for k, v in oob_preds_forest.items()})  
        y_oob = y.loc[oob_preds.index]
        self.oob_score_ += self._get_oob_score(y_oob, oob_preds, self.oob_score)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """ Прогнозирует целевую переменную.

        Args:
            X: Датасет с признаками для прогнозирования. 

        Returns:
            Предсказанные значения.

        """
        y_pred = X.apply(lambda x: self._traverse_forest(X, self.trees), axis=1)
        
        return y_pred
    
    
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
            score = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
        else:
            pass
        
        return score
    
    def _traverse_forest(self, X: pd.DataFrame, trees: list) -> float:
        """ Проход по построенному лесу. 

        Args:
            X: Датасет с признаками для прогнозирования. 
            trees: Обученные деревья.

        Returns:
            Усреднённые прогнозы деревьев.

        """
        forest_preds = []
        for tree in trees:
            tree_preds = tree.predict(X)
            forest_preds.append(tree_preds)
            
        return np.mean(forest_preds)
    
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
