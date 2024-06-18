import pandas as pd
import numpy as np

from typing import Optional, Union
from operator import __lt__, __gt__

class MyBoostClf():
    """ Класс для работы с моделью градиентного бустинга для бинарной классификации. """
    
    def __init__(self, 
                 n_estimators: int = 10, 
                 learning_rate: Union[float, callable] = 0.1, 
                 max_depth: int = 5, 
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: Optional[int] = 16,
                 reg: float = 0.1,
                 metric: Optional[str] = None,
                 max_features: Optional[float] = 0.5,
                 max_samples: Optional[float] = 0.5,
                 random_state: int = 42
                ) -> None:
        """ Конструктор для экземпляра класса MyBoostClf. 

        Args:
            n_estimators: Количество деревьев. 
            learning_rate: Скорость обучения. 
            max_depth: Максимальная глубина дерева. 
            min_samples_split: Минимальное количество наблюдений для дальнейшего ветвления. 
            max_leafs: Максимальное количество листьев в дереве. 
            bins: Количество интервалов для поиска разбиений через гистограммный метод.
            reg: Коэффициент регуляризации. 
            metric: Метрика для оценки качества модели.
            max_features: Доля признаков используемая при стохастическом обучении.
            max_samples: Доля наблюдений используемая при стохастическом обучении.
            random_state: Сид для воспроизводимости результатов.
            
        Returns:
            None

        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.metric = metric
        self.reg = reg
        
        # стохастический бустинг
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        
        # обучение
        self.trees = []
        self.leafs_cnt = 0
        self.pred_0 = None
        
        # метрики
        self.fi = None
        self.best_score = None
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            X_eval: pd.DataFrame = None,
            y_eval: pd.DataFrame = None,
            early_stopping: int = None,
            verbose: Optional[int] = None
           ) -> None:
        """ Выполняет построение модели градиентного бустинга. 

        Args:
            X: Датасет с признаками, используемый для обучения.
            y: Истинные значения целевой переменной, используемые для обучения.
            X_eval: Датасет с признаками, используемый для оценки качества модели при процедуре ранней остановки обучения. 
            y_eval: Истинные значения целевой переменной, используемые для оценки качества модели при процедуре ранней остановки обучения.
            early_stopping: Определяет допустимое количество итераций обучения, не приводящих к улучшению качества модели.
            verbose: Контролирует вывод информации о процессе обучения.

        Returns:
            None

        """
        random.seed(self.random_state)
       
        self.fi = dict.fromkeys(X.columns, 0.0)
        
        init_cols = X.columns.to_list()
        init_rows_count = X.shape[0]
        cols_sample_count = round(self.max_features * len(init_cols))
        rows_sample_count = round(self.max_samples * init_rows_count)
        
        # log_odds
        self.pred_0 = np.log(1e-15 + (np.mean(y) / (1 - np.mean(y))))
        curr_pred = pd.Series(np.full_like(y, self.pred_0, dtype=np.float64), index=y.index)
   
        if early_stopping:
            early_stop_counter = 0
            best_res = np.inf if not self.metric else -np.inf
            _compare = __lt__ if not self.metric else __gt__
            
        for i in range(1, self.n_estimators + 1):
            cols_idxs = random.sample(init_cols, cols_sample_count)
            rows_idxs = random.sample(range(init_rows_count), rows_sample_count)
            
            X_sample = X.loc[rows_idxs, cols_idxs]
            
            curr_pred_probs = self._get_probs(curr_pred)
            # обучаем дерево на антиградиенте 
            antigrad = y - curr_pred_probs
            antigrad_sample = antigrad.loc[rows_idxs]
            
            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X_sample, antigrad_sample, X.shape[0])
            
            # обновляем значения листьев
            tree._update_leafs(curr_pred_probs, antigrad, self.reg * self.leafs_cnt)
            self.leafs_cnt += tree.leafs_cnt
            self.trees.append(tree)
            
            # обновляем важность фичей
            for col in cols_idxs:
                self.fi[col] += tree.fi[col]

            # уточняем прогноз
            y_pred_log_odds = tree.predict(X)
            curr_pred += self._get_lr(i) * y_pred_log_odds       
            
            loss = self._get_loss(curr_pred, y)
            
            if verbose and i % verbose == 0:
                log = f'{i} | Loss: {loss}'
                if self.metric:
                    score = self._get_score(y, self._get_probs(y_pred_log_odds), self.metric)
                    log += f' | {self.metric}: {round(score, 2)}'

                print(log)
            
            
            # ранная остановка 
            if early_stopping:
                y_pred_eval = self.predict_proba(X_eval)
                # сравниваем метрики
                if self.metric:
                    curr_res = self._get_score(y_eval, y_pred_eval, self.metric)
                else:
                    curr_res = self._get_loss(y_pred_eval, y_eval, log_odds=False)  
                    
                if _compare(curr_res, best_res):
                    best_res = curr_res
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter == early_stopping:
                    self.best_score = best_res
                    
                    if verbose:
                        print('Early stopping callback')
                        print(f'Number of trees b4 pruning: {len(self.trees)}', sep='\t')
                        print(f'Number of trees after pruning: {len(self.trees[:-early_stopping])}', sep='\t')
                    
                    self.trees = self.trees[:-early_stopping]
                    
                    return
           
        if not self.metric:
            self.best_score = loss
        else:
            y_pred_prob = self.predict_proba(X)
            self.best_score = self._get_score(y, y_pred_prob, self.metric)
    
    
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
        y_pred_prob = pd.Series(self.pred_0, index=X.index)
        
        for i, tree in enumerate(self.trees, start=1):
            tree_preds = tree.predict(X)
            y_pred_prob += self._get_lr(i) * tree_preds
        
        return self._get_probs(y_pred_prob)
    
    
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
    
    def _get_score(self, y_true: pd.Series, y_pred_prob: pd.Series, metric: str) -> float:
        """ Вычисляет значение метрики качества классификации.

        Args:
            y_true: Истинные значения целевой переменной.
            y_pred_prob: Предсказанные вероятности. 
            metric: Метрика для оценки качества.

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
            precision = self._get_score(y_true, y_pred_prob, 'precision')
            recall = self._get_score(y_true, y_pred_prob, 'recall')
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
    
    def _get_loss(self, log_odds_pred: pd.Series, y_true: pd.Series, log_odds: bool=True) -> float:
        """ Вычисляет значение функции потерь. 

        Args:
            log_odds_pred: Предсказанные логарифмы шансов или предсказанные вероятности (в случае log_odds=False).
            y_true: Истинные значения целевой переменной.
            log_odds: Определяет способ вычисления функции ошибок в зависимости от вида прогноза: логарифм шансов или вероятность.

        Returns:
            Значение функции потерь 

        """
        if log_odds:
            loss = -np.mean(y_true * log_odds_pred - np.log(1e-15 + 1 + np.exp(log_odds_pred)))
        else:
            loss = -np.mean(y_true * np.log(1e-15 + log_odds_pred) + (1 - y_true) * np.log(1e-15 + (1 - log_odds_pred)))
        
        return loss
    
    def _get_probs(self, log_odds: pd.Series) -> pd.Series:
        """ Приводит логарифмы шансов к вероятностям. 

        Args:
            log_odds: Логарифмы шансов.

        Returns:
            Вероятности.

        """
        probs = pd.Series(np.exp(log_odds) / (1 + np.exp(log_odds)), index=log_odds.index)
        
        return probs   
    
    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f'{type(self).__name__} class: {", ".join(params)}'

############   
class treeNode():
    """ Структура данных для построения решающего дерева. """
    
    def __init__(
        self,
        feature: str = None,
        threshold: float = None,
        left = None,
        right = None,
        ig: float = None,
        value: float = None,
        obj_idxs: list = None
    ) -> None:
        """ Конструктор для экземпляра класса treeNode. 

        Args:
            feature: Признак по которому проводится разбиение.
            threshold: Пороговое значение для разбиения.
            left: Левая ветвь.
            right: Правая ветвь.
            ig: Информационный прирост от разбиения.
            value: Значение в листе.
            obj_idxs: Индексы наблюдений, попавших в лист.

        Returns:
            None

        """
        # decision node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        
        self.obj_idxs = obj_idxs
        self.value = value
        
    def is_leaf_node(self) -> bool:
        """ Индикатор листа. 

        Returns:
            True если рассматриваемый узел является листом.

        """
        return self.value is not None

################ 
class MyTreeReg():
    """ Класс для работы с моделью решающего дерева для регрессии. """
    
    def __init__(
        self,
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

        self.loss_d = {
            'MSE': pd.Series.mean,
            'MAE': pd.Series.median
        }

        # обучение
        self.leafs_cnt = 0
        self.leafs_sum = 0
        self.leafs = []
        
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
        
        self.leafs_cnt = 1
        self.root = self._build_tree_recursive(X, y, curr_depth = 0)
        
    def predict(self, X: pd.DataFrame) -> np.array:
        """ Прогнозирует целевую переменную.

        Args:
            X: Датасет с признаками для прогнозирования.

        Returns:
            Предсказанные значения целевой переменной.

        """
        y_pred = X.apply(lambda x: self._traverse_tree(x, self.root), axis=1)
        
        return y_pred
        
    def _update_leafs(self, y_pred_prob: pd.Series, antigrad: pd.Series, reg_term: float) -> None:
        """ Обновляет значения в листьях построенного дерева. Используется при обучении градиентного бустинга на антиградиенте. 

        Args:
            y_pred_prob: Предсказанные вероятности.
            antigrad: Антиградиенты, использовавшиеся при обучении дерева в рамках бустинга.
            reg_term: Значение, отвечающее за регуляризацию бустинга. 

        Returns:
            None

        """
        # проходим по листьям получившегося дерева
        for leaf in self.leafs:
            # в каждом листе смотрим на объекты
            leaf_objs = leaf.obj_idxs

            numerator = np.sum(antigrad.loc[leaf_objs])
            denominator = 1e-15 + np.sum(y_pred_prob.loc[leaf_objs] * (1 - y_pred_prob.loc[leaf_objs]))
            gamma = numerator / denominator + reg_term

            leaf.value = gamma

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
            
        decision_node = treeNode(best_split['feature'], best_split['threshold'])

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
        n_samples = len(y)
        
        leaf_value = np.mean(y)
        self.leafs_sum += leaf_value
        
        leaf_node = treeNode(value = leaf_value, obj_idxs = y.index)
        self.leafs.append(leaf_node)
        
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
