class MyKMeans():
    """ Класс для работы с моделью K-средних. """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 10, n_init: int = 3, random_state: int = 42) -> None:
        """ Конструктор для экземпляра класса MyKMeans.

        Args:
            n_clusters (int): Количество кластеров.
            max_iter (int): Максимальное количество итераций для уточнения координат центроидов.
            n_init (int): Число итераций всего алгоритма. 
            random_state (int): Сид для воспроизводимости результатов.

        Returns:
            None

        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        
        self.cluster_centers_ = []
        self.inertia_ = None
        self.features_cols = None

    def fit(self, X: pd.DataFrame) -> None:
        """ Формирует кластеры.

        Args:
            X (pd.DataFrame): Датасет для кластеризации.

        Returns:
            None

        """
        np.random.seed(self.random_state)  
        self.features_cols = X.columns.to_list()
        
        # создаём n_init вариантов кластеризации
        clusters_versions = self._get_clusters_versions(X, self.n_init, self.max_iter)
            
        # выбираем лучшую кластеризацию: ищем вариант с наименьшим WCSS
        min_wcss = np.inf
        best_clusters_version = None

        for clusters_version in clusters_versions:
            curr_wcss = self._get_wcss(X, clusters_version)
            if curr_wcss < min_wcss:
                best_clusters_version = (clusters_version, curr_wcss)
                min_wcss = curr_wcss
        
        # сохраняем лучший вариант кластеризации
        self.cluster_centers_ = [centroid for centroid in best_clusters_version[0].values()]
        self.inertia_ = best_clusters_version[1]

        X.drop(columns='curr_cluster', inplace=True)
        
    def predict(self, X: pd.DataFrame) -> list:
        """ Присваивает кластеры, используется при инференсе. 

        Args:
            X (pd.DataFrame): Датасет для кластеризации.

        Returns:
            list: Список номеров кластеров.

        """
        final_clusters = {cluster: centroid for cluster, centroid in enumerate(self.cluster_centers_)}
        clusters = self._assign_cluster(X, final_clusters)
        
        return clusters 

    def _get_clusters_versions(self, X: pd.DataFrame, n_init: int, max_iter: int) -> list:
        """ Формирует n_init вариантов кластеризации.

        Args:
            X (pd.DataFrame): Датасет для кластеризации.
            n_init (int): Число итераций всего алгоритма. 
            max_iter (int): Максимальное количество итераций для уточнения координат центроидов.

        Returns:
            list: Список вариантов кластеризации

        """
        # здесь храним варианты кластеризации
        clusters_versions = []
        
        for _ in range(n_init):
            # здесь храним кластеры и координаты центроидов
            clusters = dict.fromkeys(range(self.n_clusters), None)
            
            # инициализируем координаты центроидов 
            for cluster in range(self.n_clusters):
                coordinates = []
                for col in self.features_cols:
                    point = np.random.uniform(X[col].min(), X[col].max())
                    coordinates.append(point)
        
                clusters[cluster] = coordinates
                
            # уточняем координаты max_iter раз
            for _ in range(max_iter):
                X['curr_cluster'] = self._assign_cluster(X, clusters)
                # новые координаты
                clusters_upd = dict.fromkeys(range(self.n_clusters), None)
                for cluster in range(self.n_clusters):
                    X_cluster = X[X['curr_cluster'] == cluster][self.features_cols]
                    # если в данной итерации в кластер не попал ни один объект то оставляем старые координаты
                    if X_cluster.shape[0] == 0:
                        clusters_upd[cluster] = clusters[cluster]
                    else:
                        updated_centroid = X_cluster.mean(axis=0)
                        clusters_upd[cluster] = updated_centroid.to_list()
                # координаты центроидов не изменились с предыдущей итерации
                if clusters_upd == clusters:
                    break
                # иначе --- обновляем координаты
                else:
                    clusters = clusters_upd

            clusters_versions.append(clusters)

        return clusters_versions 

    def _assign_cluster(self, X: pd.DataFrame, clusters: dict) -> list:
        """ Распределяет наблюдения по кластерам, используется при обучении. 

        Args:
            X (pd.DataFrame): Датасет для кластеризации.
            clusters (dict): Словарь, определяющий кластеры.

        Returns:
            list: Список кластеров.

        """
        curr_clusters = []
        
        for idx in X.index:
            obj = X.loc[idx, self.features_cols]
            cluster_distances = []
            for cluster in clusters:
                distance = self._get_distance(obj, clusters[cluster])
                cluster_distances.append(distance)

            curr_cluster = np.argmin(cluster_distances)
            curr_clusters.append(curr_cluster)

        return curr_clusters

    def _get_distance(self, obj: pd.Series, centroid: list) -> float:
        """ Вычисляет расстояние между точкой (наблюдением) и центроидом кластера. 

        Args:
            obj (pd.Series): Точка для кластеризации. 
            centroid (list): Список, содержащий координаты центроида.

        Returns:
            float: Расстояние между точкой и центроидом. 

        """
        distances = np.sqrt(np.sum(np.square(obj - centroid)))
        
        return distances 
    
    def _get_wcss(self, X: pd.DataFrame, clusters_version: dict) -> float:
        """ Вычисляет WCSS --- сумму квадратов внутрикластерных расстояний до центроидов.

        Args:
            X (pd.DataFrame): Датасет для кластеризации. 
            clusters_version (dict): Словарь с рассматриваемым вариантом кластеризации.

        Returns:
            float: WCSS для рассматриваемого датасета и варианта кластеризации.

        """
        wcss = 0
        X['curr_cluster'] = self._assign_cluster(X, clusters_version)
        
        for cluster in clusters_version:
            X_cluster = X[X['curr_cluster'] == cluster]
            for idx in X_cluster.index:
                obj = X_cluster.loc[idx, self.features_cols]
                value = np.square(self._get_distance(obj, clusters_version[cluster]))
                wcss += value
        
        return wcss
    
    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            str: Строка с текстовым представлением объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f'{type(self).__name__} class: {", ".join(params)}'
