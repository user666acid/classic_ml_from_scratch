import pandas as pd
import numpy as np

class MyKNNClf():
    """ Класс для работы с моделью k ближайших соседей для бинарной классификации. """
    
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
        
    def predict(self, X: pd.DataFrame) -> np.array:
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
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
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
