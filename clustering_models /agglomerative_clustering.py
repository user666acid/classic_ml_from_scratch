import pandas as pd
import numpy as np

class MyAgglomerative():
    """ Класс для работы с моделью иерархической агломеративной кластеризации. """
    
    def __init__(self, n_clusters: int = 3, metric: str = 'euclidean') -> None:
        """ Конструктор для экземпляра класса MyAgglomerative.

        Args:
            n_clusters: Количество кластеров.
            metric: Используемая мера расстояния между точками.

        Returns:
            None

        """
        self.n_clusters = n_clusters 
        self.metric = metric

        self.clusters = None
        
    def fit_predict(self, X: pd.DataFrame) -> list:
        """ Формирует и присваивает кластеры.

        Args:
            X: Датасет для кластеризации.

        Returns:
            Номера кластеров.

        """
        #  инициализируем кластеры: сначала каждая точка определяет отдельный кластер
        self.clusters = self._init_clusters(X)

        # объединяем ближийшие между собой точки до тех пор, пока не останется заданное количество кластеров
        while len(self.clusters) > self.n_clusters:
            # находим две ближайшие между собой точки (кластер тоже может быть точкой)
            closest_clusters, points = self._find_closest_clusters()
            # обновляем кластеры 
            self.clusters = self._update_clusters(X, closest_clusters, points)

        # возвращаем итоговые кластеры 
        return [key for key, value in self.clusters.items() for _ in value[0]]
        
    def _init_clusters(self, X: pd.DataFrame) -> dict:
        """ Инициализирует множество кластеров.

        Args:
            X: Датасет для кластеризации.

        Returns:
            Словарь, определяющий кластеры, e.g.:

            {cluster_0: ([индексы объектов в кластере 0], [координаты центроида кластера 0],
             cluster_1: ([индексы объектов в кластере 1], [координаты центроида кластера 1]}

        """
        init_clusters = {idx: ([X.index[idx]], values) for idx, values in enumerate(X.to_numpy())}
        
        return init_clusters
        
    def _find_closest_clusters(self) -> tuple:
        """ Выполняет поиск двух ближайших кластеров во множестве кластеров. 

        Returns:
            Массив содержащий информацию о ближайших кластерах, e.g.:

            ((индекс перевого кластера, индекс второго кластера), (индексы точек, входящих в эти кластеры))

        """
        min_distance = np.inf
        closest_clusters = None

        clusters_ids = list(self.clusters)

        for i, cluster_i in enumerate(clusters_ids[:-1]):
            for j, cluster_j in enumerate(clusters_ids[i + 1:]):
                distance = self._get_distance(self.clusters[cluster_i][1], self.clusters[cluster_j][1])
                
                if distance < min_distance:
                    closest_clusters = (cluster_i, cluster_j)
                    points = self.clusters[cluster_i][0] + self.clusters[cluster_j][0]
                    min_distance = distance

        return closest_clusters, points
            
    def _update_clusters(self, X: pd.DataFrame, clusters_to_merge: tuple, points: list) -> dict:
        """ Обновляет множество кластеров, объединяя ближайшие два кластера. 

        Args:
            X: Датасет для кластеризации.
            clusters_to_merge: Номера объединяемых кластеров.
            points: Индексы точек объединяемых кластеров. 

        Returns:
            Словарь с обновленными кластерами, e.g.:

            {cluster_0: ([индексы объектов в кластере 0], [координаты центроида кластера 0],
             cluster_1: ([индексы объектов в кластере 1], [координаты центроида кластера 1]}

        """
        # собираем новое множество кластеров начиная с объединяемых (при объединении берём среднее по признакам)
        # центройды вседа считаются по всем точкам, входящим в кластер
        updated_clusters = {0: (points, np.mean(X.loc[points, :], axis=0))}

        # подтягиваем остальные существующие кластеры
        for cluster_id in self.clusters:
            # кроме тех, что были объединены:
            if cluster_id in clusters_to_merge:
                continue 
            updated_clusters[len(updated_clusters)] = self.clusters[cluster_id]

        return updated_clusters
        

    def _get_distance(self, cluster_i: np.ndarray, cluster_j: np.ndarray) -> float:
        """ Вычисляет расстояние (по мере metric) между (центроидами) кластеров i и j.

        Args:
            cluster_i, cluster_j: Координаты центроидов кластеров.

        Returns:
            Расстояние между кластерами.
            
        """     
        if self.metric == 'euclidean':
            distance = np.sqrt(np.sum(np.square(cluster_i - cluster_j)))
            
        elif self.metric == 'manhattan':
            distance = np.sum(np.abs(cluster_i - cluster_j))
            
        elif self.metric == 'chebyshev':
            distance = np.max(np.abs(cluster_i - cluster_j))
            
        elif self.metric == 'cosine':
            distance = 1 - (cluster_i * cluster_j).sum() \
                / (np.sqrt(np.sum(np.square(cluster_i))) * np.sqrt(np.sum(np.square(cluster_j))))
            
        else:
            raise ValueError(f'Metric {metric} is not supported')
            
        return distance
        
    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f'{type(self).__name__} class: {", ".join(params)}'
