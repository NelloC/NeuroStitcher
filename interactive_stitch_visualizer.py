import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from connector import NeuronConnector
from dash import no_update
from dash.dependencies import Input, Output


class StitchVisualizer:
    def __init__(self, neuron_data, candidates, stitches, connector: NeuronConnector):
        self.neuron_data = neuron_data
        self.candidates = candidates
        self.stitches = stitches
        self.connector = connector  # NeuronConnector per gestire lo stitching
        self.candidate_keys = list(candidates.keys())
        self.current_candidate_index = 0
        self.manual_mode_active = False
        self.selected_manual_points = []  # Per i punti selezionati manualmente

        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.translations = {
            "EN": {
                "accept": "Accept", "reject": "Reject", "next": "Next", "prev": "Previous", "manual": "Manual Stitching",        
                "ending_point": "Ending Point", "neighbor_point": "Neighbor Point", "score": "Score","attractor_points": "Attractor Points",
                "activate_clusters": "Activate Clusters", "deactivate_clusters": "Deactivate Clusters",
                "activate_growth": "Activate Growth & Direction Vectors", "deactivate_growth": "Deactivate Growth & Direction Vectors",
                "activate_attractors": "Activate Attractor Points", "deactivate_attractors": "Deactivate Attractor Points", 
                "soma": "Soma", "axon": "Axon", "dendrite": "Dendrite", "apical": "Apical Dendrite", "ending": "Ending Point", "neighbor": "Neighbor Point", "stitch":"Stitching",
                "growth_vector": "Ending Growth Vector", "neighbor_growth_vector":"Neighbor Growth Vector", "direction_vector": "Direction Vector", "attractor_points":"Attractor Points", "cluster":"Cluster", "attractor_line":"attractor_line",
                "accept_message": "Stitch Accepted. See Next.",
                "reject_message": "Stitch Rejected. See Next or Proceed Manually.",
                "manual_message": "Select points manually on the graph and Accept.",
                "cursor":"Manual Neighbor",
                "manual_off_message":"Deactivate Manual Mode. Stitch Accepted. See Next.",
                "manual_deactivate":"Manual mode disabled. Click Next",
                "selection_message": "You have selected the following points:\nEnding: {ending}\nNeighbor: {neighbor}"


            },
            "IT": {
                "accept": "Accetta", "reject": "Rifiuta", "next": "Avanti", "prev": "Indietro", "manual": "Curcitura Manuale",
                "ending_point": "Punto Finale", "neighbor_point": "Punto Vicino", "score": "Punteggio","attractor_points": "Punti Attrattori",
                "activate_clusters": "Attiva Clusters", "deactivate_clusters": "Disattiva Clusters",
                "activate_growth": "Attiva Vettori di Crescita e Direzione", "deactivate_growth": "Disattiva Vettori di Crescita e Direzione",
                "activate_attractors": "Attiva Punti Attrattori", "deactivate_attractors": "Disattiva Punti Attrattori", 
                "soma": "Soma", "axon": "Assone", "dendrite": "Dendrite", "apical": "Dendrite Apicale", "ending": "Punto Finale", "neighbor": "Punto Vicino", "stitch":"Cucitura",
                "growth_vector": "Vettore di Crescita del Punto Finale", "neighbor_growth_vector":"Vettore di Crescita del Punto Vicino", "direction_vector": "Vettore di Direzione", "attractor_points":"Punti Attrattori", "cluster":"Cluster", "attractor_line":"attractor_line",
                "accept_message": "Stitch Accettato. Vai Avanti.",
                "reject_message": "Stitch Rifiutato. Vedi il prossimo o procedi manualmente.",
                "manual_message": "Seleziona i punti manualmente sul grafico e Accetta.",
                "cursor":"Vicino Manuale",
                "manual_off_message":"Disattivare Modalità Manuale. Stitch Accettato. Vai Avanti.",
                "manual_deactivate":"Modalità Manuale disattivata. Clicca Avanti",
                "selection_message": "Hai selezionato i seguenti punti:\nPunto finale: {ending}\nPunto Vicino: {neighbor}"

            },
            "ES": {
                "accept": "Aceptar", "reject": "Rechazar", "next": "Siguiente", "prev": "Anterior", "manual": "Puntadas Manual",
                "ending_point": "Punto Final", "neighbor_point": "Punto Vecino", "score": "Puntuación", "attractor_points": "Puntos Atractores",
                "activate_clusters": "Activar Clusters", "deactivate_clusters": "Desactivar Clusters",
                "activate_growth": "Activar Vectores de Crecimiento y Direciòn", "deactivate_growth": "Desactivar Vectores de Crecimiento y Direciòn",
                "activate_attractors": "Activar Puntos Atractor", "deactivate_attractors": "Desactivar Puntos Atractor", 
                "soma": "Soma",  "axon": "Axón", "dendrite": "Dendrita", "apical": "Dendrita Apical", "ending": "Punto Final", "neighbor": "Punto Vecino", "stitch":"Puntadas",
                "growth_vector": "Vector de Crescimiento de Punto Final", "neighbor_growth_vector":"Vector de Crescimiento de Punto Vecino", "direction_vector": "Vector de Dirección", "attractor_points":"Puntos Actractor", "cluster":"Cluster" , "attractor_line":"attractor_line",
                "accept_message": "Stitch Aceptado. Ver el Siguiente.",
                "reject_message": "Stitch Rechazado. Ver el siguiente o proceder manualmente.",
                "manual_message": "Selecciona puntos manualmente en el gráfico y Acepta.",
                "cursor":"Punto Vecino manual",
                "manual_off_message":"Desactivar el Modo Manual. Stitch Aceptado. Ver el Siguiente.",
                "manual_deactivate":"Modo manual deshabilitado. Haga clic en Siguiente",
                "selection_message": "Has seleccionado los siguientes puntos:\nPunto Final: {ending}\nPunto Vecino: {neighbor}"

            }
        }

        self.app.layout = html.Div([
            html.Div([
                html.Img(src='/assets/logo.png', style={'height': '50px', 'marginRight': '10px'}),
                html.H2("NeuronStitcher", style={'display': 'inline-block', 'verticalAlign': 'middle'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),  # Contenitore flessibile per allineare immagine e titolo

            html.H4("A novel tool for neuron reconstruction by Antonello Conelli", style={'marginTop': '5px'}),
            html.Label("Select Language:"),
            dcc.RadioItems(
                id='language-selector',
                options=[
                    {'label': '🇬🇧 English', 'value': 'EN'},
                    {'label': '🇮🇹 Italiano', 'value': 'IT'},
                    {'label': '🇪🇸 Español', 'value': 'ES'}
                ],
                value='EN',
                inline=True
            ),
            dcc.Graph(
                id='neuron-plot',
                style={'height': '800px', 'width': '100%'}  # Altezza aumentata e larghezza adattata
            ),
            html.Div(id='candidate-info', style={'marginTop': 10}),
            html.Div([
                html.Button(id='toggle-clusters-btn', n_clicks=0, style={'backgroundColor': 'purple', 'color': 'white'}),
                html.Button(id='toggle-growth-btn', n_clicks=0, style={'backgroundColor': 'purple', 'color': 'white'}),
                html.Button(id='toggle-attractors-btn', n_clicks=0, style={'backgroundColor': 'purple', 'color': 'white'}),
                html.Button('Manual Stitch', id='manual-btn', n_clicks=0, style={'backgroundColor': 'blue', 'color': 'white'})
            ], style={'marginTop': 10, 'display': 'flex', 'gap': '10px'}),
            html.Div([
                html.Button(id='prev-btn', n_clicks=0),
                html.Button(id='next-btn', n_clicks=0),
                html.Button(id='accept-btn', n_clicks=0, style={'backgroundColor': 'green', 'color': 'white'}),
                html.Button(id='reject-btn', n_clicks=0, style={'backgroundColor': 'red', 'color': 'white'}),
            ], style={'marginTop': 10, 'display': 'flex', 'gap': '10px'}),

        ])

        self.setup_callbacks()

    def setup_callbacks(self):
        # Callback per aggiornare etichette pulsanti
        self.app.callback(
            Output('prev-btn', 'children'),
            Output('next-btn', 'children'),
            Output('accept-btn', 'children'),
            Output('reject-btn', 'children'),
            Output('manual-btn', 'children'),
            Output('toggle-clusters-btn', 'children'),
            Output('toggle-growth-btn', 'children'),
            Output('toggle-attractors-btn', 'children'),
            Input('language-selector', 'value'),
            Input('toggle-clusters-btn', 'n_clicks'),
            Input('toggle-growth-btn', 'n_clicks'),
            Input('toggle-attractors-btn', 'n_clicks')
        )(self.update_button_labels)
        """
        # Callback per catturare eventi tastiera
        self.app.callback(
            Output('key-event', 'data'),
            Input('keyboard-input-active', 'data')
        )(self.toggle_keyboard_input)
        """
        # Callback unificato
        self.app.callback(
            Output('neuron-plot', 'figure'),
            Output('candidate-info', 'children'),
            Input('next-btn', 'n_clicks'),
            Input('prev-btn', 'n_clicks'),
            Input('accept-btn', 'n_clicks'),
            Input('reject-btn', 'n_clicks'),
            Input('manual-btn', 'n_clicks'),
            Input('neuron-plot', 'clickData'),
            Input('toggle-clusters-btn', 'n_clicks'),
            Input('toggle-growth-btn', 'n_clicks'),
            Input('toggle-attractors-btn', 'n_clicks'),
            State('language-selector', 'value')
        )(self.unified_callback)

    """
    def toggle_keyboard_input(n_clicks):

      #  Gestisce l'attivazione e la disattivazione dell'input da tastiera.

        if n_clicks and n_clicks % 2 == 1:
            return True  # Input attivo
        return False  # Input disattivo

    def capture_key_event(self, key_data):
        
        #Mappa i tasti premuti ai pulsanti Dash corrispondenti.

        key_mapping = {
            'accept-btn': 'accept-btn',
            'reject-btn': 'reject-btn',
            'next-btn': 'next-btn',
            'prev-btn': 'prev-btn'
        }
        print(f"[DEBUG] Key Event Captured: {key_data}")
        return key_mapping.get(key_data, no_update)

    """
    def update_button_labels(self, language, clusters_clicks, growth_clicks, attractors_clicks):
        toggle_clusters_label = self.translations[language]["activate_clusters"] if clusters_clicks % 2 == 0 else self.translations[language]["deactivate_clusters"]
        toggle_growth_label = self.translations[language]["activate_growth"] if growth_clicks % 2 == 0 else self.translations[language]["deactivate_growth"]
        toggle_attractors_label = self.translations[language]["activate_attractors"] if attractors_clicks % 2 == 0 else self.translations[language]["deactivate_attractors"]

        return (
            self.translations[language]['prev'],
            self.translations[language]['next'],
            self.translations[language]['accept'],
            self.translations[language]['reject'],
            self.translations[language]['manual'],
            toggle_clusters_label,
            toggle_growth_label,
            toggle_attractors_label
        )

    def render_candidate_plot(self, candidate, show_clusters, show_attractors, show_growth, language, accepted_stitch=None):
        piece_point = candidate["ending"][:3]
        neighbor_point = candidate["candidate"][:3]
        growth_vector = candidate["growth_vector"][:3]
        direction_vectors = [vec[:3] for vec in candidate["direction_vectors"]]
        attractors = np.array(candidate["attractors"])[:, :3]
        neighbor_growth_vector = candidate["neighbor_growth_vector"][:3]

        fig = self.plot_full_neuron(language)

        # Aggiungi sempre il punto finale (Ending) al grafico
        fig.add_scatter3d(
            x=[piece_point[0]], y=[piece_point[1]], z=[piece_point[2]],
            mode='markers', marker=dict(size=9, color='blue', symbol='circle'),
            name=self.translations[language]['ending_point']
        )

        # Aggiungi sempre il punto candidato (Neighbor)
        fig.add_scatter3d(
            x=[neighbor_point[0]], y=[neighbor_point[1]], z=[neighbor_point[2]],
            mode='markers', marker=dict(size=9, color='red', symbol='circle'),
            name=self.translations[language]['neighbor_point']
        )

        if accepted_stitch:
            fig.add_scatter3d(
                x=[piece_point[0], neighbor_point[0]],
                y=[piece_point[1], neighbor_point[1]],
                z=[piece_point[2], neighbor_point[2]],
                mode='lines', line=dict(color='red', width=5),
                name="Accepted Stitch"
            )

        if show_growth:
            growth_arrow = np.array([piece_point, piece_point + 100.5 * growth_vector])
            fig.add_scatter3d(
                x=growth_arrow[:, 0], y=growth_arrow[:, 1], z=growth_arrow[:, 2],
                mode='lines', line=dict(color='orange', width=8),
                name=self.translations[language]['growth_vector']
            )

            neighbor_growth_arrow = np.array([neighbor_point, neighbor_point + 100.5 * neighbor_growth_vector])
            fig.add_scatter3d(
                x=neighbor_growth_arrow[:, 0], y=neighbor_growth_arrow[:, 1], z=neighbor_growth_arrow[:, 2],
                mode='lines', line=dict(color='orange', width=8),
                name=self.translations[language]['neighbor_growth_vector']
            )

            for i, vector in enumerate(direction_vectors):
                dir_arrow = np.array([piece_point, piece_point + 100.5 * vector])
                fig.add_scatter3d(
                    x=dir_arrow[:, 0], y=dir_arrow[:, 1], z=dir_arrow[:, 2],
                    mode='lines', line=dict(color='green', width=8),
                    name=f"{self.translations[language]['direction_vector']} {i+1}"
                )

        if show_attractors:
            for i, attractor in enumerate(attractors):
                # Punto attractor
                fig.add_scatter3d(
                    x=[attractor[0]], y=[attractor[1]], z=[attractor[2]],
                    mode='markers', marker=dict(size=9, color='purple', symbol='diamond'),
                    name=f"{self.translations[language]['attractor_points']} {i+1}"
                )
                # Linea che connette l'attractor all'ending
                fig.add_scatter3d(
                    x=[piece_point[0], attractor[0]],
                    y=[piece_point[1], attractor[1]],
                    z=[piece_point[2], attractor[2]],
                    mode='lines', line=dict(color='purple', width=5, dash='dash'),
                    showlegend=False  # Evita di aggiungere un'ulteriore voce nella legenda
                )

        if show_clusters:
            unique_labels = np.unique(candidate["cluster_labels"])
            for label in unique_labels:
                cluster_points = self.neuron_data.points[candidate["cluster_labels"] == label]
                fig.add_scatter3d(
                    x=cluster_points[:, 0], y=cluster_points[:, 1], z=cluster_points[:, 2],
                    mode='markers', marker=dict(size=3, color=px.colors.qualitative.Set1[label % len(px.colors.qualitative.Set1)]),
                    name=f"{self.translations[language]['cluster']} {label}"
                )
        if self.manual_mode_active and len(self.selected_manual_points) == 2:
            neighbor_point = self.selected_manual_points[1]
            fig.add_scatter3d(
                x=[neighbor_point[0]], y=[neighbor_point[1]], z=[neighbor_point[2]],
                mode='markers', marker=dict(size=9, color='red', symbol='circle'),
                name=self.translations[language]['neighbor_point']
            )

        return fig




    def update_plot(self, next_clicks, prev_clicks, accept_clicks, reject_clicks, manual_clicks, 
                    toggle_clusters, toggle_growth, toggle_attractors, language):
        """
        Aggiorna il grafico e le informazioni del candidato corrente in base all'azione dell'utente.
        """
        if not self.candidate_keys:
            return self.plot_full_neuron(language), "No candidates available."

        trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        accepted_stitch = None
        message = ""
        message_color = "black"  # Default message color

        # Navigazione tra i candidati
        if trigger == 'next-btn':
            self.current_candidate_index = (self.current_candidate_index + 1) % len(self.candidate_keys)
        elif trigger == 'prev-btn':
            self.current_candidate_index = (self.current_candidate_index - 1) % len(self.candidate_keys)
        elif trigger == 'accept-btn':
            if self.manual_mode_active and len(self.selected_manual_points) == 2:
                # Aggiungi il candidato manuale
                new_candidate = {
                    'ending': self.selected_manual_points[0],
                    'candidate': self.selected_manual_points[1],
                    'score': 0.0,  # Manual stitches have no score
                    'attractors': []  # No attractors in manual mode
                }
                self.connector.connect_components([new_candidate])
                self.candidates[f"manual_{len(self.candidates)}"] = [new_candidate]
                self.candidate_keys.append(f"manual_{len(self.candidates) - 1}")
                message = self.translations[language]['manual_off_message']
                message_color = "blue"
                self.selected_manual_points = []  # Reset dei punti manuali
            else:
                # Accettazione normale
                candidate_key = self.candidate_keys[self.current_candidate_index]
                current_candidate = self.candidates[candidate_key][0]
                self.connector.connect_components([current_candidate])  # Salva lo stitch
                message = self.translations[language]['accept_message']
                message_color = "green"
                self.stitches.append(current_candidate)
        elif trigger == 'reject-btn':
            candidate_key = self.candidate_keys[self.current_candidate_index]
            current_candidate = self.candidates[candidate_key][0]

            # Rimuovi la connessione di stitching
            self.connector.remove_connection(current_candidate)

            message = self.translations[language]['reject_message']
            message_color = "red"
            for stitch in self.stitches:
                if np.array_equal(stitch, current_candidate):
                    self.stitches.remove(stitch)
                    break

        # Mostra il candidato attuale con possibili aggiornamenti
        candidate_key = self.candidate_keys[self.current_candidate_index]
        current_candidate = self.candidates[candidate_key][0]

        candidate_info = self.format_candidate_info(current_candidate, language)

        # Combina il messaggio e le informazioni in un unico output HTML
        combined_info = html.Div([
            html.P(message, style={"fontWeight": "bold", "color": message_color}),
            html.Pre(candidate_info)  # Usa Pre per un formato più leggibile
        ])

        return (
            self.render_candidate_plot(current_candidate, toggle_clusters, toggle_attractors, toggle_growth, language, accepted_stitch),
            combined_info
        )




    def format_candidate_info(self, candidate, language):
        attractor_details = "\n".join([f"  - {point}" for point in candidate['attractors']])

        info_text = (
            f"{self.translations[language]['ending_point']}: {candidate['ending']}\n"
            f"{self.translations[language]['neighbor_point']}: {candidate['candidate']}\n"
            f"{self.translations[language]['score']}: {candidate['score']:.4f}\n"
            #f"{self.translations[language]['attractor_points']}: {len(candidate['attractors'])}:\n"
            #f" \n{attractor_details}"
        )
        return info_text

    def plot_full_neuron(self, language):
        """
        Visualizza il neurone completo con linee colorate in base al tipo (soma, dendriti, assone).
        """
        points = self.neuron_data.points
        lines = self.neuron_data.lines
        type_to_id_mapping = self.neuron_data.get_type_to_id_mapping()
        object_properties = self.neuron_data.get_object_properties()

        # Colori per ogni tipo di struttura
        type_colors = {
            type_to_id_mapping["soma"]: 'darkorange',  # Soma
            type_to_id_mapping["dendrite"]: 'green',   # Dendriti
            type_to_id_mapping["axon"]: 'blue',        # Assone
            type_to_id_mapping["apical"]: 'purple',    # Apical dendrite
            type_to_id_mapping["stitch"]: 'red',
        }

        # Etichette leggibili per la legenda
        type_labels = {v: k.capitalize() for k, v in type_to_id_mapping.items()}
        # Traduzioni leggenda
        type_labels = {
            v: self.translations[language][k.lower()]
            for k, v in type_to_id_mapping.items()
        }
        plotted_types = set()  # Per evitare duplicati nella legenda

        fig = go.Figure()

        for line in lines:
            line_type = line[0]
            start, length = line[1], line[2]
            line_points = points[start: start + length]

            show_legend = line_type not in plotted_types
            plotted_types.add(line_type)

            fig.add_scatter3d(
                x=line_points[:, 0], y=line_points[:, 1], z=line_points[:, 2],
                mode='lines',
                line=dict(color=type_colors.get(line_type, 'gray'), width=3),
                name=type_labels.get(line_type, 'Unknown') if show_legend else None,
                showlegend=show_legend
            )

        return fig


    def start(self):
        self.app.run_server(debug=True)


    def unified_callback(self, next_clicks, prev_clicks, accept_clicks, reject_clicks, manual_clicks, click_data,
                        toggle_clusters_clicks, toggle_growth_clicks, toggle_attractors_clicks, language):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Priorità agli eventi da tastiera
        """
        if key_event:
            trigger = key_event  # Sovrascrive il trigger se esiste un evento tastiera
        print(f"[DEBUG] Trigger: {trigger}, Manual Mode: {self.manual_mode_active}")
        """
        # Controllo degli stati dei toggle
        show_clusters = toggle_clusters_clicks % 2 != 0
        show_growth = toggle_growth_clicks % 2 != 0
        show_attractors = toggle_attractors_clicks % 2 != 0

        # Azioni principali in base ai trigger
        if trigger in ['next-btn', 'prev-btn', 'accept-btn', 'reject-btn']:
            return self.update_plot(
                next_clicks, prev_clicks, accept_clicks, reject_clicks, manual_clicks,
                show_clusters, show_growth, show_attractors, language
            )

        if trigger == 'manual-btn':
            self.manual_mode_active = not self.manual_mode_active
            message = (self.translations[language]['manual_message']
                    if self.manual_mode_active else self.translations[language]['manual_deactivate'])

            # Quando la modalità manuale è attiva, evidenzia l'ending corrente in blu
            if self.manual_mode_active:
                current_candidate = self.candidates[self.candidate_keys[self.current_candidate_index]][0]
                ending_point = current_candidate['ending'][:3]

                fig = self.plot_full_neuron(language)
                fig.add_scatter3d(
                    x=[ending_point[0]], y=[ending_point[1]], z=[ending_point[2]],
                    mode='markers', marker=dict(size=10, color='blue', symbol='circle'),
                    name=self.translations[language]['ending_point']
                )
                return fig, html.P(message, style={"color": "blue"})
            else:
                return self.plot_full_neuron(language), html.P(message, style={"color": "blue"})

        if trigger == 'neuron-plot' and self.manual_mode_active:
            if click_data:
                point = (
                    click_data['points'][0]['x'],
                    click_data['points'][0]['y'],
                    click_data['points'][0]['z']
                )
                self.selected_manual_points = [
                    self.candidates[self.candidate_keys[self.current_candidate_index]][0]['ending'], 
                    point
                ]

                fig = self.render_candidate_plot(
                    self.candidates[self.candidate_keys[self.current_candidate_index]][0],
                    show_clusters=False, show_attractors=True, show_growth=False, language=language
                )
                ending_point = self.selected_manual_points[0][:3]
                fig.add_scatter3d(
                    x=[ending_point[0]], y=[ending_point[1]], z=[ending_point[2]],
                    mode='markers', marker=dict(size=10, color='blue', symbol='circle'),
                    name=self.translations[language]['ending_point']
                )
                fig.add_scatter3d(
                    x=[point[0]], y=[point[1]], z=[point[2]],
                    mode='markers', marker=dict(size=10, color='yellow', symbol='circle'),
                    name=self.translations[language]['cursor']
                )
                selection_message = self.translations[language]['selection_message'].format(
                    ending=ending_point,
                    neighbor=point
                )

                return fig, html.Div([
                    html.P(selection_message, style={"color": "blue", "fontWeight": "bold"}),
                    html.Pre(f"Ending: {ending_point}\nNeighbor: {point}")
                ])

        if trigger == 'accept-btn' and self.manual_mode_active:
            print("[DEBUG] Inside manual accept")
            if len(self.selected_manual_points) == 2:
                new_candidate = {
                    'ending': self.selected_manual_points[0],
                    'candidate': self.selected_manual_points[1],
                    'score': 0.0,
                    'attractors': []
                }
                self.connector.connect_components([new_candidate])
                self.candidates[f"manual_{len(self.candidates)}"] = [new_candidate]
                self.candidate_keys.append(f"manual_{len(self.candidates) - 1}")
                self.stitches.append(new_candidate)
                return self.plot_full_neuron(language), html.P(self.translations[language]['manual_off_message'], style={"color": "blue"})

        # Default: aggiorna il plot con i toggle
        return self.update_plot(
            next_clicks, prev_clicks, accept_clicks, reject_clicks, manual_clicks,
            show_clusters, show_growth, show_attractors, language
        )







