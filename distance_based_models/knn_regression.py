import pandas as pd
import numpy as np

class MyKNNReg():
    """ Класс для работы с моделью k ближайших соседей для регрессии. """
    
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
