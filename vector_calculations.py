import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from storage import NeuronDataStorage  # Assicurati di avere la classe NeuronDataStorage importata


class VectorCalculator:
    def calculate_growth_vector(self, points, smoothing_method='adaptive', sigma=1.0, window_size=10, **kwargs):
        if points.shape[0] < 2:  # PCA richiede almeno 2 punti con variazione
            #print("[DEBUG] Punti insufficienti per PCA.")
            return np.zeros(3)

        # Debug punti originali
        #print(f"[DEBUG] Punti originali prima dello smoothing: {points}")
        
        # Applica lo smoothing
        smoothed_points = self.apply_smoothing(points, method=smoothing_method, sigma=sigma, window_size=window_size)
        
        # Debug punti smoothed
        #print(f"[DEBUG] Punti smoothed: {smoothed_points}")
        
        if smoothed_points.shape[0] < 2:  # PCA richiede almeno 2 punti con variazione anche dopo lo smoothing
            #print("[DEBUG] Punti insufficienti dopo lo smoothing.")
            return np.zeros(3)

        # Calcolo del growth vector tramite PCA
        pca = PCA(n_components=2)
        try:
            pca.fit(smoothed_points)
            growth_vector = pca.components_[0]
            #print(f"[DEBUG] Growth Vector calcolato: {growth_vector}")
            return growth_vector[:3] #/ np.linalg.norm(growth_vector) if np.linalg.norm(growth_vector) != 0 else np.zeros(3)
        except Exception as e:
            print(f"[ERROR] Errore durante il calcolo PCA: {e}")
            return np.zeros(3)
    
    def calculate_angle_between_vectors(self, v1, v2):
        # Assicurati che entrambi i vettori abbiano dimensioni 3
        v1 = v1[:3] if len(v1) > 3 else v1
        v2 = v2[:3] if len(v2) > 3 else v2
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0  # Restituisce 0 gradi se uno dei vettori è nullo

        angle = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
        return np.degrees(angle)

    def calculate_direction_vector(self, point, neighbor_point):
        direction_vector = np.array(neighbor_point) - np.array(point)
        norm = np.linalg.norm(direction_vector)
        return direction_vector / norm if norm != 0 else np.zeros(3)

    def calculate_curvature_change(self, growth_vector, neighbor_growth_vector):
        # Assicurati che entrambi i vettori abbiano 3 dimensioni
        growth_vector = growth_vector[:3]  # Prendi solo le prime 3 componenti
        neighbor_growth_vector = neighbor_growth_vector[:3]

        norm_growth_vector = growth_vector / np.linalg.norm(growth_vector) if np.linalg.norm(growth_vector) != 0 else np.zeros(3)
        norm_neighbor_growth_vector = neighbor_growth_vector / np.linalg.norm(neighbor_growth_vector) if np.linalg.norm(neighbor_growth_vector) != 0 else np.zeros(3)
        
        dot_product = np.dot(norm_growth_vector, norm_neighbor_growth_vector)
        return np.arccos(np.clip(dot_product, -1.0, 1.0))


    # Funzioni ausiliarie per smoothing
    def apply_smoothing(self, points, method='adaptive', sigma=1.0, window_size=3):
        if method == 'adaptive':
            return self.adaptive_smoothing(points)
        elif method == 'spline':
            return self.spline_smoothing(points)
        elif method == 'weighted':
            return self.weighted_moving_average(points, window_size)
        else:
            return self.smooth_points(points, sigma)
    
    def weighted_moving_average(self, points, window_size=3):
        weights = np.arange(1, window_size + 1)
        smoothed_points = np.copy(points)
        for i in range(points.shape[1]):
            smoothed_points[:, i] = np.convolve(points[:, i], weights / weights.sum(), mode='valid')
        return smoothed_points

    def spline_smoothing(self, points, s=0):
        smoothed_points = np.zeros_like(points)
        for i in range(points.shape[1]):
            spline = UnivariateSpline(np.arange(points.shape[0]), points[:, i], s=s)
            smoothed_points[:, i] = spline(np.arange(points.shape[0]))
        return smoothed_points

    def adaptive_smoothing(self, points, min_sigma=0.5, max_sigma=2.0):
        if points.shape[0] < 3:
            #print("[DEBUG] Punti insufficienti per lo smoothing, restituisco i punti originali.")
            return points

        # Calcolo della curvatura e del valore dinamico per sigma
        curvatures = np.gradient(np.gradient(points, axis=0), axis=0)
        curvature_magnitude = np.linalg.norm(curvatures, axis=1)

        if np.allclose(curvature_magnitude.max(), curvature_magnitude.min()):
            #print("[DEBUG] Curvature costante, nessun smoothing adattivo applicato.")
            return points  # Nessuna variazione significativa

        norm_curvature = (curvature_magnitude - curvature_magnitude.min()) / (curvature_magnitude.max() - curvature_magnitude.min())
        adaptive_sigma = min_sigma + (max_sigma - min_sigma) * norm_curvature

        smoothed_points = np.zeros_like(points)
        for i in range(points.shape[1]):  # Loop per ogni dimensione (x, y, z)
            smoothed_points[:, i] = [
                gaussian_filter1d(points[:, i], sigma=sigma_value)[idx] 
                for idx, sigma_value in enumerate(adaptive_sigma)
            ]

        return smoothed_points


    def smooth_points(self, points, sigma=1.0):
        smoothed_points = np.zeros_like(points)
        for i in range(points.shape[1]):
            smoothed_points[:, i] = gaussian_filter1d(points[:, i], sigma=sigma)
        return smoothed_points

    def filter_out_noise(self, points):
        mean = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        return points[np.all(np.abs(points - mean) < 2 * std_dev, axis=1)]

