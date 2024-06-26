import pandas as pd
import numpy as np

class MyDBSCAN():
    """ Класс для работы с моделью кластеризации DBSCAN."""
    
    def __init__(self, eps: float = 3, min_samples: int = 3, metric: str = 'euclidean') -> None:
        """ Конструктор для экземпляра класса MyDBSCAN.

        Args:
            eps: Радиус поиска.
            min_samples: Минимальное число соседей для корневой точки.
            metric: Используемая мера расстояния между точками.

        Returns:
            None
        
        """
        
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        self.distance_matrix = None
        self.labels = None
    
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Формирует и присваивает кластеры.

        Args: 
            X: Датасет для кластеризации.

        Returns:
            Номера кластеров.
        
        """
        N = X.shape[0]
        self.distance_matrix = self._get_distance_matrix(X)

        self.labels = np.zeros(N)
        self.cluster_id = 0

        # проходим по объектам выборки и отбираем корневые точки
        for i in range(N):
            # не рассматриваем уже посещённые точки
            if not (self.labels[i] == 0):
                continue 

            # находим соседей i-ой точки
            neighbors = self._get_neighbors(i)

            # является ли точка выбросом 
            if len(neighbors) < self.min_samples + 1:
                self.labels[i] = -1
            
            # иначе --- начинаем формировать кластер
            else:
                self.cluster_id += 1
                self._grow_cluster(i, neighbors)

        return self.labels 

    def _get_distance_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """ Вычисляет попарные расстояния между объектами обучающей выборки.
        
        Args:
            X: Датасет для кластеризации.
            
        Returns:
            Матрица попарных расстояний.

        """
        distance_matrix = np.zeros((X.shape[0], X.shape[0]))
        
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                X_i, X_j = X.iloc[i, :], X.iloc[j, :]
                distance = self._get_distance(X_i, X_j)
                distance_matrix[i, j] = distance
        
        return distance_matrix

    def _get_distance(self, point_i: pd.Series, point_j: pd.Series) -> float:
        """ Вычисляет расстояние (по мере self.metric) между точками point_i и point_j.

        Args:
            point_i, point_j: Точки (наблюдения) исходного датасета.

        Returns:
            Расстояние между точками.
            
        """
        if self.metric == 'euclidean':
            distance = np.sqrt(np.sum(np.square(point_i - point_j)))
        elif self.metric == 'manhattan':
            distance = np.sum(np.abs(point_i - point_j))
        elif self.metric == 'chebyshev':
            distance = np.max(np.abs(point_i - point_j))
        elif self.metric == 'cosine':
            distance = 1 - (point_i * point_j).sum() \
                / (np.sqrt(np.sum(np.square(point_i))) * np.sqrt(np.sum(np.square(point_j))))
        else:
            raise ValueError(f'Metric {metric} is not supported')
            
        return distance
        
    def _grow_cluster(self, root_idx: int, neighbors: np.ndarray) -> None:
        """ Выполняет процедуру формирования кластера вокруг корневой точки root_idx.

        Args:
            root_idx: Индекс корневой точки, вокруг которой формируется кластер.
            neighbors: Индексы точек-соседей.

        Returns:
            None
            
        """
        # i-ая точка --- корневая, присваиваем кластер
        self.labels[root_idx] = self.cluster_id

        # проходим по соседям корневой точки
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            # если сосед помечен как потенциальный выброс, то он не может быть корневой точкой -> граничная точка, просто назначаем кластер
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = self.cluster_id

            # иначе --- назначаем кластер и рассматриваем уже его соседей 
            elif self.labels[neighbor] == 0:
                self.labels[neighbor] = self.cluster_id
                new_neighbors = self._get_neighbors(neighbor)
                
                if len(new_neighbors) >= self.min_samples + 1:
                    neighbors = np.append(neighbors, new_neighbors)

            j += 1

    def _get_neighbors(self, point_idx: int) -> np.ndarray:
        """ Выполняет поиск соседей (в радиусе eps) для точки point_idx.

        Args:
            point_idx: Индекс точки для которой необходимо найти соседей.

        Returns:
            Индексы точек-соседей.

        """
        neighbors = np.argwhere(self.distance_matrix[point_idx] < self.eps).flatten()

        return neighbors
    
    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f'{type(self).__name__} class: {", ".join(params)}'
