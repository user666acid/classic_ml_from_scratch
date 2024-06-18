import pandas as pd
import numpy as np

class MyPCA():
    """ Класс для работы с методом главных компонент. """
    
    def __init__(self, n_components: int = 3) -> None:
        """ Конструктор для экземпляра класса MyPca. 

        Args:
            n_components: Количество главных компонент.

        Returns:
            None

        """
        self.n_components = n_components
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Выполняет снижение размерности исходного датасета.

        Args:
            X: Исходный датасет.

        Returns:
            Преобразованный датасет пониженной размерности.

        """
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        values, vectors = np.linalg.eigh(cov_matrix)  
        idxs = np.argsort(values)[-self.n_components:][::-1]
        principal_components = vectors[:, idxs]
        
        X_transformed = X_centered @ principal_components
         
        return X_transformed
    
    def __repr__(self) -> str:
        """ Выводит текстовое представление объекта класса.

        Returns:
            Текстовое представление объекта класса.

        """
        params = []
        for k, v in self.__dict__.items():
            params.append(f'{k}={v}')
            
        return f'{type(self).__name__} class: {", ".join(params)}'
