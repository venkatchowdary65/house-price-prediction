import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers between coordinate arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        coords = X[["Latitude", "Longitude"]].to_numpy()
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.kmeans_.fit(coords)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        safe_rooms = df["AveRooms"].replace(0, np.nan)
        safe_occup = df["AveOccup"].replace(0, np.nan)
        safe_pop = df["Population"].replace(0, np.nan)

        df["rooms_per_person"] = (df["AveRooms"] / safe_occup).fillna(0.0)
        df["bedroom_ratio"] = (df["AveBedrms"] / safe_rooms).fillna(0.0)
        df["pop_per_household"] = (df["Population"] / safe_occup).fillna(0.0)
        df["income_per_room"] = (df["MedInc"] / safe_rooms).fillna(0.0)
        df["income_per_person"] = ((df["MedInc"] * 100_000) / safe_pop).fillna(0.0)

        df["dist_sf_km"] = haversine_km(
            df["Latitude"], df["Longitude"], 37.7749, -122.4194
        )
        df["dist_la_km"] = haversine_km(
            df["Latitude"], df["Longitude"], 34.0522, -118.2437
        )
        df["min_dist_hub_km"] = np.minimum(df["dist_sf_km"], df["dist_la_km"])

        df["geo_cluster"] = self.kmeans_.predict(df[["Latitude", "Longitude"]])
        return df