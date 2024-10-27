import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from storage import NeuronDataStorage  # Assicurati di avere la classe NeuronDataStorage importata

class VectorCalculator:
    def calculate_growth_vector(self, neuron_data: NeuronDataStorage, line_idx=None, sigma=1.0, smoothing_method='adaptive', window_size=10, n_components=2):
        """
        Calcola il vettore di crescita usando la PCA sui punti smoothed.
        """
        points = neuron_data.points if line_idx is None else neuron_data.points[line_idx]

        if points.shape[0] < 2:
            return np.zeros(3)
        
        points = self.filter_out_noise(points)
        points = self.apply_smoothing(points, smoothing_method, sigma, window_size)
        significant_points = points[-window_size:]
        
        pca = PCA(n_components=n_components)
        pca.fit(significant_points)
        growth_vector = pca.components_[0]
        
        # Calcola direzione media
        avg_direction = np.mean(np.diff(significant_points, axis=0), axis=0)
        avg_direction /= np.linalg.norm(avg_direction) if np.linalg.norm(avg_direction) != 0 else 1
        
        return growth_vector / np.linalg.norm(growth_vector) if np.linalg.norm(growth_vector) != 0 else np.zeros(3)

    def calculate_angle_between_vectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        return np.degrees(angle)

    def calculate_direction_vector(self, point, neighbor_point):
        direction_vector = np.array(neighbor_point) - np.array(point)
        norm = np.linalg.norm(direction_vector)
        return direction_vector / norm if norm != 0 else np.zeros(3)

    def calculate_curvature_change(self, growth_vector, neighbor_growth_vector):
        norm_growth_vector = growth_vector / np.linalg.norm(growth_vector)
        norm_neighbor_growth_vector = neighbor_growth_vector / np.linalg.norm(neighbor_growth_vector)
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
        curvatures = np.gradient(np.gradient(points, axis=0), axis=0)
        curvature_magnitude = np.linalg.norm(curvatures, axis=1)
        norm_curvature = (curvature_magnitude - curvature_magnitude.min()) / (curvature_magnitude.max() - curvature_magnitude.min())
        adaptive_sigma = min_sigma + (max_sigma - min_sigma) * norm_curvature

        smoothed_points = np.zeros_like(points)
        for i in range(points.shape[1]):
            smoothed_points[:, i] = gaussian_filter(points[:, i], sigma=adaptive_sigma[i])
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

