from neuronmorphology_v06 import NeuronMorphology
import numpy as np
import re
import matplotlib.pyplot as plt
import networkx as nx  # Assicurati di importare networkx
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
import threading
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from functools import lru_cache
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class NeuronStitcher(NeuronMorphology):
    def __init__(self,neuronDict,unitOrientationOrigin):
        super().__init__(neuronDict,unitOrientationOrigin)
        
    def getEndPoint(self,ending):
        lineId,atEnd = ending
        line = self.lines[lineId]
        return self.points[line[1]+atEnd*(line[2]-1),0:3]

    def getEndingsByPieceId(self, pieces, includeStart=True):
        # Return all terminal points of a given piece
        def getTerminals(lineId, piece):
            terminals = []
            if isinstance(piece, list):
                for subPiece in piece:
                    terminals.extend(getTerminals(lineId, subPiece))
            else:
                terminals.append((lineId, 1))
            return terminals
    
        endingsByPieceId = {0: [(0, 0)]}  # Root ending
    
        # Se `pieces` è un set, bisogna iterare direttamente sugli ID dei pezzi
        for pieceId in pieces:
            if 0 <= pieceId < len(self.lines):
                piece = self.lines[pieceId]  # Ottieni il pezzo basato su pieceId
                endingsByPieceId[pieceId] = getTerminals(pieceId, piece)
                if includeStart:
                    endingsByPieceId[pieceId].insert(0, (pieceId, 0))
            else:
                print(f"Warning: Piece with ID {pieceId} is out of bounds in self.lines.")
    
        return endingsByPieceId


    def getSectionByLineId(self,lineIds,asNumber = False):
        objectProps = self.getObjectProperties()
        if asNumber:
            allSectionIds = set()
            for objectId,props in objectProps.items():
                if 'sectionId' in props:
                     allSectionIds.add(props['sectionId'])
                     
            # apply natural sorting
            convert = lambda s: int(s) if s.isdigit() else s.lower()
            alphanum_key = lambda key: [convert(part) for part in re.split('([0-9]+)', key)]
            indexBySectionId = { key:i for i,key in enumerate( sorted(allSectionIds, key=alphanum_key) ) }
            
        sectionByLineId = {}
        for lineId in lineIds:
            line = self.lines[lineId]
            firstPointId = line[1]
            try:
                sectionId = objectProps[firstPointId]['sectionId']
                if asNumber:
                    sectionId = indexBySectionId[sectionId]
                sectionByLineId[lineId] = sectionId
            except:
                pass
        return sectionByLineId

    """
    Estimate soma location from dendrite start points
    """
    def approximateSomaLocation(self):
        axonType = 2
        dendriteType = 3
        apicalType = 4
        dendriteStartPointIds = []
        for line in self.lines:
            if line[0] == dendriteType or line[0] == apicalType:
                dendriteStartPointIds.append(line[1])
        dendriteStartPoints = self.points[dendriteStartPointIds,0:3]
        dendriteStartCenter = np.median(dendriteStartPoints,axis=0)
        return dendriteStartCenter


    def verifyParentIndices(self):
        errors = []
        for idx, line in enumerate(self.lines):
            parentIdx = line[3]  # Supponendo che l'indice del genitore sia nella posizione 3
            if parentIdx < 0 or parentIdx >= len(self.lines):
                errors.append((idx, parentIdx, "Parent index out of range"))
            elif parentIdx == idx:
                continue  # Self-reference is allowed
            elif parentIdx != -1:  # -1 could mean no parent
                parentLine = self.lines[parentIdx]
                if not parentLine:
                    errors.append((idx, parentIdx, "Parent line does not exist"))
        return errors

    def verifyReverseProperty(self):
        errors = []
        for idx, line in enumerate(self.lines):
            if 'reverse' in line and not isinstance(line['reverse'], bool):
                errors.append((idx, line['reverse'], "Invalid reverse property"))
        return errors

    def verifyAll(self):
        parentErrors = self.verifyParentIndices()
        reverseErrors = self.verifyReverseProperty()
        
        if parentErrors:
            for error in parentErrors:
                print(f"Line {error[0]} has invalid parent index {error[1]}: {error[2]}")
        else:
            print("All parent indices are valid.")

        if reverseErrors:
            for error in reverseErrors:
                print(f"Line {error[0]} has invalid reverse property {error[1]}: {error[2]}")
        else:
            print("All reverse properties are valid.")
            
    def calculateAngleBetweenVectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)

    def calculateDirectionVector(self, point, neighbor_point):
        endPoint = np.array(point)
        neighbor_point = np.array(neighbor_point)
        
        # Calcola il vettore di direzione
        directionVector = neighbor_point - endPoint
        norm = np.linalg.norm(directionVector)
        
        # Controlla se la distanza è zero
        if norm == 0:
            #print(f"Attenzione: il punto finale ({endPoint}) e il punto di attrattore ({neighbor_point}) sono identici.")
            return np.zeros(3) # Restituisce un vettore nullo per evitare errori
        
        # Normalizza il vettore di direzione
        directionVector /= norm
        
        return directionVector

    """
    def smooth_points(self, points, sigma=1.0):

        if points.shape[0] < 2:
            return points
        
        # Filtro gaussiano sui punti (si presume che points sia un array 2D)
        smoothed_points = np.copy(points)
        for i in range(points.shape[1]):
            smoothed_points[:, i] = gaussian_filter(points[:, i], sigma=sigma)
        return smoothed_points
    """

    def calculateGrowthVector(self, lineIdx, sigma=1.0, n_components=2, smoothing_method='adaptive', window_size=10):
        currentLine = self.lines[lineIdx]
        endIdx = currentLine[1] + currentLine[2]
        points = self.points[:endIdx, 0:3]
        
        # Verifica se ci sono abbastanza punti
        if points.shape[0] < 2:
            print(f"Warning: Line {lineIdx} has too few points. Using fallback.")
            growthVector = points[-1] - points[0] if points.shape[0] == 2 else np.zeros(3)
            norm = np.linalg.norm(growthVector)
            return growthVector / norm if norm != 0 else np.zeros(3)
        
        # Filtro del rumore
        points = self.filter_out_noise(points)
        
        # Smoothing dei punti
        if smoothing_method == 'adaptive':
            points = self.adaptive_smoothing(points)  # Utilizza lo smoothing adattivo basato sulla curvatura
        elif smoothing_method == 'spline':
            points = self.spline_smoothing(points)
        elif smoothing_method == 'weighted':
            points = self.weighted_moving_average(points, window_size=window_size)
        else:
            points = self.smooth_points(points, sigma=sigma)  # Gaussian smoothing come fallback
        
        # Seleziona gli ultimi 10 punti o meno se la linea è più corta
        significant_points = points[-window_size:]  # Seleziona fino agli ultimi 10 punti
        
        # Applica la PCA sui punti selezionati
        pca = PCA(n_components=n_components)
        pca.fit(significant_points)
        
        # Il vettore di crescita è la prima componente principale
        growth_vector = pca.components_[0]
        
        # Calcola la direzione media dei punti
        avg_direction = np.mean(np.diff(significant_points, axis=0), axis=0)
        avg_direction /= np.linalg.norm(avg_direction) if np.linalg.norm(avg_direction) != 0 else 1
        
        # Calcola l'angolo tra il vettore di crescita e la direzione media
        angle = self.calculateAngleBetweenVectors(growth_vector, avg_direction)
        
        norm = np.linalg.norm(growth_vector)
        return growth_vector / norm if norm != 0 else np.zeros(3)


    
    
    # Metodi ausiliari per smoothing e filtri
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
            smoothed_points[:, i] = gaussian_filter(points[:, i], sigma=adaptive_sigma[i])  # Usa solo la componente corretta per ogni asse
        return smoothed_points

    
    def smooth_points(self, points, sigma=1.0):
        smoothed_points = np.zeros_like(points)
        for i in range(points.shape[1]):  # Applica il filtro lungo ogni asse separatamente
            smoothed_points[:, i] = gaussian_filter1d(points[:, i], sigma=sigma)
        return smoothed_points
    
    def is_better_growth_vector(self, new_vector, old_vector):
        # Implementa una logica per determinare se il nuovo vettore di crescita è migliore del precedente
        return np.linalg.norm(new_vector) > np.linalg.norm(old_vector)  # Placeholder
    
    def filter_out_noise(self, points):
        mean = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        filtered_points = points[np.all(np.abs(points - mean) < 2 * std_dev, axis=1)]
        return filtered_points

        

    def estimate_eps(self, points, k=20):
        # Calcola le distanze ai k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(points)
        distances, indices = neighbors_fit.kneighbors(points)
        
        # Prendi la distanza al k-esimo vicino più vicino
        k_distances = distances[:, k-1]
        
        # Ordina le distanze in modo crescente
        k_distances_sorted = np.sort(k_distances)
        
        # Trova il "ginocchio" nella curva delle distanze (elbow method)
        # Puoi usare un approccio visivo o un algoritmo per trovare il ginocchio
        # Qui, ad esempio, prendi un percentile alto come stima dell'eps
        eps_estimate = np.percentile(k_distances_sorted, 10)  # Prendi il 90-esimo percentile
        
        return eps_estimate
        
        
    def estimate_density_threshold(self, labels, counts, percentile=50):
        """
        Stima un density_threshold dinamico basato sulle dimensioni dei cluster
        :param labels: Le etichette restituite da DBSCAN
        :param counts: Il numero di punti per cluster
        :param percentile: Il percentile da utilizzare per stabilire la soglia
        :return: density_threshold dinamico
        """
        # Escludi i cluster con meno di 2 punti (considerati outlier)
        cluster_sizes = counts[labels[labels >= 0]]  # Filtro i cluster validi
    
        if len(cluster_sizes) == 0:
            return 0  # Nessun cluster rilevato
    
        # Calcola un percentile delle dimensioni dei cluster come soglia di densità
        density_threshold = np.percentile(cluster_sizes, percentile)
    
        return density_threshold



    def run_dbscan(self, points, min_samples=2, k=20, percentile=25):
        """
        Esegue DBSCAN con parametri eps e density_threshold stimati dinamicamente.
        
        :param points: Lista di coordinate dei punti su cui eseguire DBSCAN.
        :param min_samples: Numero minimo di punti in un cluster.
        :param k: Numero di vicini usato per stimare eps.
        :param percentile: Percentile per stimare la soglia di densità.
        
        :return: labels, unique_labels, counts, tipo_struttura
        """
        # Stima eps dinamicamente
        eps_estimate = self.estimate_eps(points, k=k)
        print(f"Eps stimato: {eps_estimate}")
        
        # Esegui DBSCAN con il valore stimato di eps
        dbscan = DBSCAN(eps=eps_estimate, min_samples=min_samples).fit(points)
        labels = dbscan.labels_
        
        # Conta i punti in ogni cluster
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        # Debug: stampa il numero totale di cluster e outliers
        print(f"Numero di cluster identificati: {len(unique_labels[unique_labels >= 0])}")  # Escludi outliers
        print(f"Numero di outliers: {np.sum(labels == -1)}")
        
        # Stima dinamicamente il density_threshold
        density_threshold_estimate = self.estimate_density_threshold(labels, counts, percentile=percentile)#np.median(counts)
        print(f"Density threshold stimato: {density_threshold_estimate}")
        
        # Determina se la densità è alta o bassa in base alla soglia
        avg_cluster_size = np.mean(counts[counts > 1]) if len(counts[counts > 1]) > 0 else 0
        print(f"Densità media calcolata: {avg_cluster_size}")
        
        if avg_cluster_size > density_threshold_estimate:
            tipo_struttura = 'densa'
            print(f"Zona densa rilevata con densità media: {avg_cluster_size}")
            
        else:
            tipo_struttura = 'sparse'
            print(f"Zona poco densa rilevata con densità media: {avg_cluster_size}")
        
        return labels, unique_labels, counts, tipo_struttura



    def get_adaptive_params(self, cluster_points, avg_cluster_size, labels, unique_labels, counts, tipo_struttura, default_min_samples=2):
        """
        Funzione adattiva per regolare num_neighbors e distance_factor in base alla densità locale dei punti.
        """
        # Usa run_dbscan per ottenere la struttura dei cluster
        #labels, unique_labels, counts, tipo_struttura = self.run_dbscan(cluster_points, min_samples=default_min_samples)
    
        # Stima la densità dinamicamente usando la dimensione media del cluster
        density_threshold = self.estimate_density_threshold(labels, counts)#np.median(counts)
    
        # Parametri di default
        num_neighbors = default_min_samples
        distance_factor = 0.0  # Fattore di default
    
        # Adattamento per alta densità (sopra la soglia stimata)
        if avg_cluster_size >= density_threshold:
            num_neighbors = max(10, default_min_samples - int((avg_cluster_size - density_threshold) / density_threshold * 3))
            distance_factor = max(0.6, 1.5 - (avg_cluster_size - density_threshold) / density_threshold * 0.5)
    
        # Adattamento per bassa densità (sotto la soglia stimata)
        elif avg_cluster_size < density_threshold:
            num_neighbors = min(20, default_min_samples + int((density_threshold - avg_cluster_size) / density_threshold * 5))
            distance_factor = min(1.5, 1.0 + (density_threshold - avg_cluster_size) / density_threshold * 0.5)
    
        # Garantisci che num_neighbors e distance_factor siano nei limiti definiti
        num_neighbors = min(20, max(10, num_neighbors))  # Limita il numero di vicini tra 5 e 10
        distance_factor = min(1.5, max(0.6, distance_factor))  # Limita il distance_factor tra 0.8 e 1.5
    
        return num_neighbors, distance_factor




    
    
    def calculateAttractorPoints(self, pieces, densityThreshold=100):
        """
        Calcola i punti di attrazione per ogni ending, con parametri adattivi per num_neighbors e distance_threshold 
        calcolati individualmente per ciascun ending, tenendo conto degli ending circostanti di tutti i PieceId.
        
        :param pieces: Lista dei pieceId su cui calcolare i punti attrattori.
        :param densityThreshold: Soglia di densità per considerare un cluster ad alta densità.
        
        :return: Attractor points, numero di attractor points, growthVectors, neighborGrowthVectors, DirectionVectors, distances, proportions, pieceAttractorPointsMapping, distance_factor.
        """
        attractorPoints = {}
        self.attractorPointsCount = {}
        distance_factors = {}
        
        endingsByPieceId = self.getEndingsByPieceId(pieces)
        
        all_end_points = []
        point_piece_map = []
        ending_indices_map = []
        
        for pieceId, endings in endingsByPieceId.items():
            for idx, ending in enumerate(endings):
                point = self.getEndPoint(ending)
                if point is not None:
                    all_end_points.append(point)
                    point_piece_map.append(pieceId)
                    ending_indices_map.append(idx)
                else:
                    print(f"Skipping null endpoint for pieceId {pieceId}, ending {idx}")
        
        if not all_end_points:
            print("No terminal points found. KDTree not built.")
            return attractorPoints, {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        # Esegui DBSCAN una sola volta
        all_end_points_np = np.array(all_end_points)
        labels, unique_labels, counts, tipo_struttura = self.run_dbscan(all_end_points_np)
        
        # Debug: informazioni sui cluster identificati da DBSCAN
        #print(f"DBSCAN found {len(unique_labels)} clusters. Unique labels: {unique_labels}")
        #for label, count in zip(unique_labels, counts):
            #print(f"Cluster {label} has {count} points.")
        
        kdtree = KDTree(all_end_points_np)
        
        growthVectors = {}
        neighborGrowthVectors = {}
        DirectionVectors = {}
        distances = {}
        pieceAttractorPointsMapping = {}
        
        for pieceId in pieces:
            pieceAttractorPoints = set()
            
            if pieceId not in growthVectors:
                growthVectors[pieceId] = []
            if pieceId not in neighborGrowthVectors:
                neighborGrowthVectors[pieceId] = []
            if pieceId not in DirectionVectors:
                DirectionVectors[pieceId] = []
            if pieceId not in distances:
                distances[pieceId] = []
            if pieceId not in pieceAttractorPointsMapping:
                pieceAttractorPointsMapping[pieceId] = []
            if pieceId not in distance_factors:
                distance_factors[pieceId] = []  # Inizializza come lista per ogni pieceId
            
            for ending_idx, ending in enumerate(endingsByPieceId[pieceId]):
                point = self.getEndPoint(ending)
                if point is not None:
                    if len(growthVectors[pieceId]) <= ending_idx:
                        growthVector = self.calculateGrowthVector(pieceId)
                        growthVectors[pieceId].append(growthVector)
                    else:
                        growthVector = growthVectors[pieceId][ending_idx]
                    
                    point_index = next(i for i, p in enumerate(all_end_points) if np.array_equal(p, point))
                    cluster_id = labels[point_index]
            
                    if cluster_id != -1:  # Ignora gli outliers
                        cluster_points = all_end_points_np[labels == cluster_id]
                        cluster_index = np.where(unique_labels == cluster_id)[0][0]
                        avg_cluster_size = counts[cluster_index]
    
                        num_neighbors, distance_factor = self.get_adaptive_params(cluster_points, avg_cluster_size, labels, unique_labels, counts, tipo_struttura)
                        distance_factors[pieceId].append(distance_factor)
                    else:
                        num_neighbors, distance_factor = 10, 0.6
                        distance_factors[pieceId].append(distance_factor)
                    
                    distances_to_neighbors, indices = kdtree.query(point, k=num_neighbors)
                    
                    for idx, distance in zip(indices, distances_to_neighbors):
                        if idx >= len(all_end_points) or idx < 0:
                            print(f"Skipping invalid index {idx} for pieceId {pieceId}")
                            continue
                        
                        neighbor_point = all_end_points[idx]
                        neighbor_pieceId = point_piece_map[idx]
                        neighbor_ending_index = ending_indices_map[idx]
                        
                        if neighbor_pieceId != pieceId and not np.array_equal(point, neighbor_point):
                            if neighbor_pieceId not in growthVectors:
                                growthVectors[neighbor_pieceId] = []
                            if neighbor_pieceId not in neighborGrowthVectors:
                                neighborGrowthVectors[neighbor_pieceId] = []
                            
                            if len(growthVectors[neighbor_pieceId]) <= neighbor_ending_index:
                                neighborGrowthVector = self.calculateGrowthVector(neighbor_pieceId)
                                growthVectors[neighbor_pieceId].append(neighborGrowthVector)
                            else:
                                neighborGrowthVector = growthVectors[neighbor_pieceId][neighbor_ending_index]
                            
                            directionVector = self.calculateDirectionVector(point, neighbor_point)
                            
                            pieceAttractorPoints.add((tuple(neighbor_point), neighbor_pieceId, neighbor_ending_index))
                            neighborGrowthVectors[pieceId].append(neighborGrowthVector)
                            DirectionVectors[pieceId].append(directionVector)
                            distances[pieceId].append(distance)


                    
            attractorPoints[f'{pieceId}'] = list(pieceAttractorPoints)
            pieceAttractorPointsMapping[pieceId] = pieceAttractorPoints
        
        self.attractorPointsCount[pieceId] = len(attractorPoints.get(pieceId, []))
        
        # Calcola le densità per i punti di attrazione
        density_mapping = {}
        
        for pieceId in pieces:
            if pieceId in attractorPoints:
                density_mapping[pieceId] = len(attractorPoints[pieceId]) / sum(distance_factors[pieceId]) if distance_factors[pieceId] else 0
        
        # Ordina i punti di attrazione per densità
        # Ordina i punti di attrazione per densità
        sorted_attractor_points = {k: v for k, v in sorted(attractorPoints.items(), 
                                                               key=lambda item: density_mapping.get(item[0], 0), 
                                                               reverse=True)}

        
        # Crea una mappatura per mantenere l'ordine originale di ritorno
        # Crea una mappatura per mantenere l'ordine originale di ritorno
        sorted_pieceAttractorPointsMapping = {k: pieceAttractorPointsMapping.get(k, []) for k in sorted_attractor_points.keys()}

        
        return sorted_attractor_points, self.attractorPointsCount, growthVectors, neighborGrowthVectors, DirectionVectors, distances, sorted_pieceAttractorPointsMapping, distance_factors


    def adapt_weights_based_on_dbscan_density(self, points, labels, unique_labels, counts, tipo_struttura, percentile = 50):
        # Esegui DBSCAN per ottenere le informazioni sui cluster
        #labels, unique_labels, counts, tipo_struttura = self.run_dbscan(points)
        
        # Calcola la densità media dei cluster
        avg_cluster_size = np.mean(counts[counts > 1]) if len(counts[counts > 1]) > 0 else 0
        density_threshold_estimate = self.estimate_density_threshold(labels, counts, percentile=percentile) #np.median(counts)
        # Adatta i pesi dinamicamente in base alla densità
        if avg_cluster_size > density_threshold_estimate:  # Se densità alta
            w_dist = 0.2  # Peso basso alla distanza
            w_growth = 0.1  # Peso maggiore agli angoli di crescita
            w_dir = 0.3  # Peso moderato agli angoli di direzione
            w_cur = 0.4
        else:  # Se densità bassa
            w_dist = 0.3  # Peso maggiore alla distanza
            w_growth = 0.1  # Peso minore agli angoli di crescita
            w_dir = 0.2  # Peso minore agli angoli di direzione
            w_cur = 0.4
        
        return w_dist, w_growth, w_dir, w_cur
    
    # Usa questa funzione all'interno della funzione di calcolo del punteggio
    
    def calculate_score_with_sigmoid(self, points, distances, growth_angles, direction_angles, curvature_change, labels, unique_labels, counts, tipo_struttura):
        # Adatta i pesi basandosi sulla densità calcolata tramite DBSCAN
        w_dist, w_growth, w_dir, w_cur = self.adapt_weights_based_on_dbscan_density(points, labels, unique_labels, counts, tipo_struttura)
    

    
        # Normalizza le distanze e gli angoli
        normalized_distance = 1 / (1 + np.exp(-distances))
        normalized_growth_angle = 1 / (1 + np.exp(-growth_angles))
        normalized_direction_angle = 1 / (1 + np.exp(-direction_angles))
        normalized_curvature_change = 1 / (1 + np.exp(-curvature_change))

    
    
        # Calcola il punteggio totale
        score = (w_dist * normalized_distance +
                 w_growth * normalized_growth_angle +
                 w_dir * normalized_direction_angle +
                 w_cur * normalized_curvature_change)
    
        # Stampa il valore del punteggio calcolato
        #print(f"Score: {score}")
    
        return score



    def calculate_curvature_change(self, growth_vector, neighbor_growth_vector):
        """
        Calcola la variazione angolare tra due growth vector consecutivi.
        
        :param growth_vector: Vettore di crescita corrente.
        :param neighbor_growth_vector: Vettore di crescita del neurite vicino.
        :return: Il cambiamento di curvatura (variazione angolare in radianti).
        """
        # Normalizza i growth vector per evitare problemi di scaling
        norm_growth_vector = growth_vector / np.linalg.norm(growth_vector)
        norm_neighbor_growth_vector = neighbor_growth_vector / np.linalg.norm(neighbor_growth_vector)
        
        # Calcola il prodotto scalare tra i due growth vector
        dot_product = np.dot(norm_growth_vector, norm_neighbor_growth_vector)
        
        # Limita il prodotto scalare tra -1 e 1 per evitare errori numerici
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calcola la variazione angolare (curvatura) tra i due growth vector
        curvature_change = np.arccos(dot_product)
        
        return curvature_change


    def log_debug_info(self, endingId, neighbor_pieceId, neighbor_ending_index, distance, growth_angle, direction_angle):
        print(f"Ending: {endingId}, Neighbor:{neighbor_pieceId}-{neighbor_ending_index}, "
              f"Distance: {distance}, Growth Angle: {growth_angle}, Direction Angle: {direction_angle}")        

    def findBestStitches(self, pieces, somaLocations=None, stitchesPerEndingPenalty={0: 0, 1: 0, 2: 0},
                         manualStitchLocations=None, sectionJumpPenalty={1: 0, 2: 0}, noPenaltyThreshold=0,
                         angleWeight=1.0, proportionTolerance=2.0, initial_weight_ratio=0.3):
        
        results = {}
    
        def attempt_connections_for_threshold(somaLocations):
            self.endingsByPieceId = self.getEndingsByPieceId(pieces)
            sectionByPieceId = self.getSectionByLineId([lineId for lineId in pieces], asNumber=True)
    
            if somaLocations is None:
                somaLocations = [self.approximateSomaLocation()]
            else:
                if not isinstance(somaLocations[0], (list, tuple)):
                    somaLocations = [somaLocations]
    
            attractorPoints, attractorPointsCount, growthVectors, neighborGrowthVectors, DirectionVectors, distances, pieceAttractorPointsMapping, distance_factors = self.calculateAttractorPoints(pieces)
            max_distance = max(distances[pieceId][j] for pieceId in pieces for j in range(len(distances[pieceId])))
    
            # Costruisci la lista dei punti da passare a DBSCAN
            all_end_points = []
            point_piece_map = []
            ending_indices_map = []
            
            endingsByPieceId = self.getEndingsByPieceId(pieces)
            
            for pieceId, endings in endingsByPieceId.items():
                for idx, ending in enumerate(endings):
                    point = self.getEndPoint(ending)
                    if point is not None:
                        all_end_points.append(point)
                        point_piece_map.append(pieceId)
                        ending_indices_map.append(idx)
                    else:
                        print(f"Skipping null endpoint for pieceId {pieceId}, ending {idx}")
            
            if not all_end_points:
                print("No terminal points found. DBSCAN not executed.")
                return  # O gestire diversamente in base alla logica
    
            # Esegui DBSCAN una sola volta
            all_end_points_np = np.array(all_end_points)
            labels, unique_labels, counts, tipo_struttura = self.run_dbscan(all_end_points_np)
    
            # Crea un dizionario per la densità dei pezzi
            density_by_pieceId = {}
            
            for point_index, pieceId in enumerate(point_piece_map):
                if labels[point_index] != -1:  # Escludi gli outlier
                    if pieceId not in density_by_pieceId:
                        density_by_pieceId[pieceId] = 0
                    density_by_pieceId[pieceId] += 1  # Incrementa il conteggio per questo pieceId
    
            # Ordina i pezzi in base alla densità (decrescente)
            pieces_sorted_by_density = sorted(density_by_pieceId.keys(), key=lambda pid: density_by_pieceId[pid], reverse=True)
    
            unconnectedPieces = set(pieces_sorted_by_density)  # Popola unconnectedPieces in base alla densità
            connectedPieces = set()
            receivingEndings = set()
            numStitchesPerEnding = {}
            stitches = []
            endPointPositions = {}
            self.validStitchesCount = 0
            unconnectedEndings = set()
            unconnectedPieces_lock = threading.Lock()

    
            for pieceId in pieces_sorted_by_density:
                endings = self.endingsByPieceId.get(pieceId, [])
                for i in range(len(endings)):
                    endingId = f'{pieceId}-{i}'
                    unconnectedEndings.add(endingId)
    
            def addStitch(stitchFrom, stitchTo):
                stitches.append((stitchFrom, stitchTo))
                numStitchesPerEnding[stitchFrom] = numStitchesPerEnding.get(stitchFrom, 0) + 1
                receivingEndings.add(stitchTo)
                toPieceId, toEndingIndex = map(int, stitchTo.split('-'))
                connectedPieces.add(toPieceId)
    
                with unconnectedPieces_lock:
                    if toPieceId in unconnectedPieces:
                        unconnectedPieces.remove(toPieceId)
    
                unconnectedEndings.discard(stitchFrom)
                unconnectedEndings.discard(stitchTo)
    
            def find_and_add_best_stitch(pieceId):
                best_score = None  # Cambia per trovare il punteggio minimo
                bestEndingId = None
                bestCandidateEndingId = None
                bestGrowthAngle = None
                bestDistance = None
                bestDirectionAngle = None
            
                endings = self.endingsByPieceId.get(pieceId, [])
            
                # Debug: Inizio calcolo per il pieceId
                #print(f"[DEBUG] Starting calculation for pieceId {pieceId}")

                # Calcola la distanza minima globale tra tutte le connessioni j-esime
                min_distance = float('inf')
            
                # Trova la distanza minima per tutte le connessioni j-esime per questo pieceId
                for i, ending in enumerate(endings):
                    endingId = f'{pieceId}-{i}'
                    numStitches = numStitchesPerEnding.get(endingId, 0)
            
                    if endingId not in receivingEndings and (numStitches + 1) in stitchesPerEndingPenalty:
                        for j, distance in enumerate(distances[pieceId]):
                            if distance < min_distance:
                                min_distance = distance
                                # Debug: Nuova distanza minima trovata


                    
                                # Debug: Nuova distanza minima trovata
                                #print(f"[DEBUG] Found new min distance for pieceId {pieceId}, ending {i}, neighbor {j}: {min_distance}")
            
                # Debug: Mostra la distanza minima calcolata
                #print(f"[DEBUG] PieceId: {pieceId}, Min distance: {min_distance}")
            
                # Itera sui punti di attrazione in ordine di densità
                sorted_attractor_points = attractorPoints.get(f'{pieceId}', [])
            
                for i, ending in enumerate(endings):
                    endingId = f'{pieceId}-{i}'
                    numStitches = numStitchesPerEnding.get(endingId, 0)
            
                    if endingId not in receivingEndings and (numStitches + 1) in stitchesPerEndingPenalty:
                        endPoint = self.getEndPoint(ending)
            
                        # Ottieni il distance_factor per l'ending attuale
                        distance_factors_for_ending = distance_factors[pieceId][i]
            
                        # Calcola la soglia di distanza per l'ending attuale
                        distance_threshold = min_distance + distance_factors_for_ending * min_distance

            
                        # Debug: Mostra la soglia di distanza calcolata
                        #print(f"[DEBUG] Calculated distance threshold for endingId {endingId}: {distance_threshold} (min distance: {min_distance}, factor: {distance_factors_for_ending})")
            
                        all_distances_within_threshold = True
                        any_other_points_within_threshold = False
            
                        # Verifica se tutte le distanze rientrano nella soglia
                        for j, distance in enumerate(distances[pieceId]):
                            if distance > distance_threshold:
                                all_distances_within_threshold = False
                                # Debug: Distanza fuori soglia
                                #print(f"[DEBUG] Distance {distance} is OUTSIDE the threshold {distance_threshold}")
                            else:
                                any_other_points_within_threshold = True
                                # Debug: Distanza entro la soglia
                                #print(f"[DEBUG] Distance {distance} is WITHIN the threshold {distance_threshold}")
                        
                        
                        if not all_distances_within_threshold:
                            # Debug: Non tutte le distanze sono entro la soglia
                            #print(f"[DEBUG] PieceId: {pieceId}, Not all distances are within the threshold.")
                            continue  # Salta questo ending se non tutte le distanze sono entro la soglia
                        
                        
                        #if any_other_points_within_threshold:
                            # Debug: Altri punti entro la soglia
                            #print(f"[DEBUG] PieceId: {pieceId}, There are other points within the distance threshold.")
                        
                        #print(f"[DEBUG] sorted_attractor_points for pieceId {pieceId}: {sorted_attractor_points}")


                        for j, (neighbor_point, neighbor_pieceId, neighbor_ending_index) in enumerate(sorted_attractor_points):
                        #    distance = distances[pieceId][j]
                            #if distance < distance_threshold:
                            print(f"[DEBUG] Processing sorted attractor point for pieceId {pieceId}: {neighbor_pieceId}, {neighbor_ending_index}")
                            distance = distances[pieceId][j]
                            growthVector = growthVectors[pieceId][i]
                            neighbor_growthVector = neighborGrowthVectors[pieceId][j]
                            directionVector = DirectionVectors[pieceId][j]
            
                            growth_angle = self.calculateAngleBetweenVectors(growthVector, neighbor_growthVector)
                            direction_angle = self.calculateAngleBetweenVectors(growthVector, directionVector)
                            curvature_change = self.calculate_curvature_change(growthVector, neighbor_growthVector)
                                #print(f"[DEBUG] all_end_points_np for pieceId {pieceId}: {all_end_points_np}")
                                #print(f"[DEBUG] distances for pieceId {pieceId}: {distance}")
                                #print(f"[DEBUG] growth_angle for pieceId {pieceId}: {growth_angle}")
                                #print(f"[DEBUG] direction_angle for pieceId {pieceId}: {direction_angle}")
                                #self.log_debug_info(endingId, neighbor_pieceId, neighbor_ending_index, distance, growth_angle, direction_angle)
                                
                                # Calcola il punteggio combinato
                            score = self.calculate_score_with_sigmoid(
                                    all_end_points_np, distance, growth_angle, direction_angle, curvature_change, labels, unique_labels, counts, tipo_struttura)
                           # print(f"[DEBUG] score for pieceId {pieceId}: {score}")
            
                                # Confronta il punteggio con il miglior punteggio trovato
                            if best_score is None or score < best_score:  # Cambia per ottenere il punteggio minimo
                                best_score = score
                                bestEndingId = endingId
                                bestCandidateEndingId = f"{neighbor_pieceId}-{neighbor_ending_index}"
                                bestGrowthAngle = growth_angle
                                bestDistance = distance
                                bestDirectionAngle = direction_angle
                                    

            
                # Debug: Stampa se ha trovato una connessione o no
                if best_score is not None:
                    print(f"PieceId: {pieceId}, Best connection found: {bestEndingId} to {bestCandidateEndingId} with score {best_score}")

            
                    fromPieceId, fromEndingIndex = map(int, bestEndingId.split('-'))
                    toPieceId, toEndingIndex = map(int, bestCandidateEndingId.split('-'))
                    endings_list = self.endingsByPieceId.get(toPieceId, [])
            
                    if toEndingIndex < 0 or toEndingIndex >= len(endings_list):
                        print(f"Errore: toEndingIndex {toEndingIndex} è fuori intervallo per pieceId {toPieceId}.")
                    else:
                        endPointPositions[bestEndingId] = self.getEndPoint(self.endingsByPieceId[fromPieceId][fromEndingIndex])
                        endPointPositions[bestCandidateEndingId] = self.getEndPoint(endings_list[toEndingIndex])
            
                        addStitch(bestEndingId, bestCandidateEndingId)
            
                        pieceRemoved = False
                        with unconnectedPieces_lock:
                            if pieceId in unconnectedPieces:
                                unconnectedPieces.remove(pieceId)
                                pieceRemoved = True
            
                        if pieceRemoved:
                            print(f"Connect {bestEndingId} to {bestCandidateEndingId} at growth angle {bestGrowthAngle:.2f}; "
                                  f"and direction angle {bestDirectionAngle:.2f} "
                                  f"at attractor distance {bestDistance:.2f}. "
                                  f"{len(unconnectedPieces)} to go")
            
                        return pieceId
                else:
                    print(f"Nessuna connessione trovata per {pieceId}.")
            
            """        
                # Tentativo di ripetizione per i pezzi rimasti non connessi
            remaining_unconnected_pieces = list(unconnectedPieces)
            for remaining_pieceId in remaining_unconnected_pieces:
                print(f"Tentando una nuova connessione per pieceId {remaining_pieceId} rimasto non connesso.")
                result = find_and_add_best_stitch(remaining_pieceId)
                if result:
                    print(f"Nuova connessione trovata per {remaining_pieceId}.")
                else:
                    print(f"Nessuna connessione trovata per {remaining_pieceId}.")
            """
    
            while unconnectedPieces:
                pieces_to_remove = set()
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(find_and_add_best_stitch, pieceId): pieceId for pieceId in list(unconnectedPieces)}
                    for future in as_completed(futures):
                        pieceId = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                pieces_to_remove.add(result)
                        except Exception as exc:
                            print(f"Piece {pieceId} generated an exception: {exc}")
    
                if not pieces_to_remove:
                    print("No more pieces to remove. Breaking out of loop.")
                    break
    
                # Rimuovi i pezzi connessi dalla lista unconnectedPieces
                unconnectedPieces = [pieceId for pieceId in unconnectedPieces if pieceId not in pieces_to_remove]
    
                # Riorganizza unconnectedPieces in base alla densità
                density_by_pieceId = {pieceId: density_by_pieceId[pieceId] for pieceId in unconnectedPieces}
                unconnectedPieces.sort(key=lambda pid: density_by_pieceId[pid], reverse=True)
    
                numUnconnectedEndings = len(unconnectedPieces)
                print(f"Unconnected pieces remaining: {numUnconnectedEndings}")
    
            numUnconnectedEndings = len(unconnectedPieces)
            print(f"Numero di endings non connessi: {numUnconnectedEndings}")
    
            return {
                'stitches': stitches,
                'endPointPositions': endPointPositions,
                'attractorPointsCount': attractorPointsCount,
                'numUnconnectedEndings': numUnconnectedEndings,
                'unconnectedEndings': unconnectedEndings,
                'DirectionVectors': DirectionVectors,
                'growthVectors': growthVectors
            }
    
        print("Attempting connections")
        result = attempt_connections_for_threshold(somaLocations)
    
        return result



        
    def invertLine(self, lineId, newParentLineId, line2children, line2pointIds):
        lines = self.lines
        tp, firstPointId, numPoints, parentLineId, negOffset = lines[lineId]
    
        # Debug output
        print(f"Inverting line {lineId}: firstPointId={firstPointId}, numPoints={numPoints}, parentLineId={parentLineId}")
    
        # List the points in reversed order
        pointIds = list(range(firstPointId + numPoints - 1, firstPointId - 1, -1))
        line2pointIds[lineId] = pointIds
        print(f"Reversed points for line {lineId}: {pointIds}")
    
        # Replace the parentId
        lines[lineId][3] = newParentLineId
    
        if parentLineId:
            # The last point of the old parent now becomes the last point of this line
            parentLine = lines[parentLineId]
            lastParentPointId = parentLine[1] + parentLine[2] - 1
            pointIds.append(lastParentPointId)
            parentLine[2] -= 1  # numPoints of parent line is reduced by one
            print(f"Updated parent line {parentLineId}: numPoints={parentLine[2]}")
    
            # The inverted line becomes the parent of all its siblings
            siblings = line2children[parentLineId]
            for siblingId in siblings:
                if siblingId != lineId:
                    lines[siblingId][3] = lineId  # Set parent of sibling to current line
                    print(f"Set parent of sibling line {siblingId} to {lineId}")
    
            # Now recursively invert the parent line, with the current line as its parent
            self.invertLine(parentLineId, lineId, line2children, line2pointIds)



    def rebuildCustomProperties(self,point2new,obj2new):
        propertyAssignments = self.customProps['for']
        newPropertyAssignments = []
        for index,assignment in enumerate(propertyAssignments):
            keep = False
            newAssignment = {}
            if 'points' in assignment:
                newPoints = []
                for o in assignment['points']:
                    if o in point2new:
                        newPoints.append(point2new[o])
                if len(newPoints):
                    keep = True
                    newAssignment['points'] = newPoints
            if 'objects' in assignment:
                newObjects = []
                for objId in assignment['objects']:
                    if objId in obj2new:
                        newObjects.append(obj2new[objId])
                if len(newObjects):
                    keep = True
                    newAssignment['objects'] = newObjects
            if keep:
                newAssignment['set'] = assignment['set'].copy()
                newPropertyAssignments.append(newAssignment)
        self.customProps['for'] = newPropertyAssignments
        self._objectProps = None


    def rebuildPoints(self,line2pointIds):
        # make sure line2children includes mixed-type branches
        line2children = self.getLine2children(mixTypes=True)

        #newPoints = [self.points[0,:].tolist()] # points[0] is a dummy point, indexing starts at 1
        newPointIds = [0]

        def addTree(lineId):
            line = self.lines[lineId]
            if lineId in line2pointIds:
                pointIds = line2pointIds[lineId]
            else:
                pointIds = range(line[1],line[1]+line[2])
            line[1] = len(newPointIds) # pointId of the next added point
            for p in pointIds:
                newPointIds.append(p)
            if lineId in line2children:
                for ch in line2children[lineId]:
                    addTree(ch)

        for lineId,line in enumerate(self.lines):
            if lineId and not line[3]:
                # no parent, meaning start of a tree (piece of neurite)
                addTree(lineId)
        
        self.points = self.points[newPointIds,:]
        point2new = { old:new for new,old in enumerate(newPointIds) }
        return point2new
        
    
    def applyStitches(self, pieces,stitches, connectPieces=True,maxConnectDistance=-1,markAsMarker=True,markAsNeurite=4,markAsColor=False,verbosity=0):
        # find the incoming connection for each piece
        incomingByPiece = {}
        for idx,stitch in enumerate(stitches):
            toPieceId,toEndingIndex = [int(k) for k in stitch[1].split('-')]
            #incomingByPiece[toPieceId] = fromEndingId
            if toPieceId in incomingByPiece:
                print('ERROR, piece {} already has an incoming connection.'.format(toPieceId))
            incomingByPiece[toPieceId] = idx

        # first make sure that each piece starts at the incoming ending 
        line2children = self.getLine2children(mixTypes=False)
        line2pointIds = {}
        firstpointIds = [line[1] for line in self.lines]
        for pieceId in pieces:
            if pieceId in incomingByPiece:
                # find the 'incoming ending' (the one that has an upstream connection)
                fromEndingId,toEndingId = stitches[incomingByPiece[pieceId]]
                toPieceId,toEndingIndex = [int(k) for k in toEndingId.split('-')]
                toLineId,toAtEnd = self.endingsByPieceId[pieceId][toEndingIndex]

                # if the incoming ending is not already the starting line of a piece, then invert the line
                if toAtEnd:
                    self.invertLine(toLineId,0, line2children,line2pointIds)
                    if verbosity>0:
                        print('Invert',pieceId,toLineId)

        # inverting lines has the points messed up, fix that
        point2new = self.rebuildPoints(line2pointIds)
        newLine2objIds = [line[1] for line in self.lines]
        obj2new = { firstpointId:self.lines[lineId][1] for lineId,firstpointId in enumerate(firstpointIds) }

        # after messing up the points, rebuild customProperties
        self.rebuildCustomProperties(point2new,obj2new)
        
        # create a new custom type for stitches
        if markAsMarker:
            stitchTypeId = self.addNeuriteType('marker',{'name':'stitch'})

        # now the stitches can be applied
        stitchedObjectIds = []
        unstitchedObjectIds = []
        objectProps = self.getObjectProperties()
        for pieceId in pieces:
            if pieceId in incomingByPiece:
                # find the 'incoming ending' (the one that has an upstream connection)
                fromEndingId,toEndingId = stitches[incomingByPiece[pieceId]]
                fromPieceId,fromEndingIndex = [int(k) for k in fromEndingId.split('-')]
                fromLineId,fromAtEnd = self.endingsByPieceId[fromPieceId][fromEndingIndex]
                toPieceId,toEndingIndex = [int(k) for k in toEndingId.split('-')]
                toLineId,toAtEnd = self.endingsByPieceId[pieceId][toEndingIndex]

                accept = True
                if fromLineId>0 and maxConnectDistance>0:
                    fromPoint = self.getEndPoint((fromLineId,1))
                    toPoint = self.getEndPoint((toLineId,0))
                    dst = np.linalg.norm(fromPoint-toPoint)
                    if dst>maxConnectDistance:
                        if verbosity>0:
                            print('Skip connection ',fromEndingId,toEndingId,dst)
                        accept = False

                toLine = self.lines[toLineId]
                if accept:
                    # apply the stitch by setting the parent of toLine to fromLineId
                    if connectPieces:
                        toLine[3] = fromLineId
                    stitchedObjectIds.append(toLine[1])
                else:
                    unstitchedObjectIds.append(toLine[1])

                if markAsMarker:
                    # add a marker that highlights the stitch
                    pointIdx = self.addPoint(self.points[toLine[1]])
                    self.addLine(stitchTypeId,pointIdx,1,0)

                if markAsNeurite and fromLineId>0:
                    # also add a line piece that highlights the stitch, the value of markAsNeurite is used as the neurite type (4=apical dendrite)
                    fromLine = self.lines[fromLineId]
                    fromPoint = self.points[fromLine[1]+fromLine[2]-1].copy()
                    fromPoint[3] = 12 # large radius
                    toPoint = self.points[toLine[1]].copy()
                    toPoint[3] = 3 # smaller radius to create arrow effect
                    firstPointIdx = self.addPoint(fromPoint)
                    self.addPoint(toPoint)
                    self.addLine(markAsNeurite,firstPointIdx,2,0)
                    # assign line to section
                    objId = fromLine[1]
                    props = objectProps[objId] if objId in objectProps else None 
                    if 'sectionId' in props:
                        objectProps[firstPointIdx] = { 'sectionId': props['sectionId'] }

        self.setObjectProperties(objectProps)

        if markAsColor:
            self.customProperties['for'] = [{
               'objects': unstitchedObjectIds,
               'set': {
                   'color': '#ccdd00'
               }
            },{
               'objects': stitchedObjectIds,
               'set': {
                   'color': '#0000ff'
               }
            }]



