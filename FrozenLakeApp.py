import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel, QSlider, QDoubleSpinBox, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QProgressBar, QGroupBox, QSizePolicy, QMessageBox, QFileDialog,QListWidget,QGraphicsDropShadowEffect ,QTabWidget
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve
import gymnasium as gym
import pickle
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time

class FrozenLakeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.collected_strawberries = set()
        self.custom_map_normal = [
            "SFFFFHHFFF",
            "FFFFFFFFFF",
            "FFFFFFFFFF",
            "FFFHHFFFFF",
            "FFFFFFFFFF",
            "FFFFFFFFFF",
            "FFFFFHHFFF",
            "FFFFFFFFFF",
            "FFFFFFFFFF",
            "FFFFFFFFFG",
        ]
        self.custom_map_strawberry = [
        "SFFFFHHFFF",
        "BFFBFFBFFF",
        "FFFFFFFFFF",
        "FFFHHFBFFF",
        "FFFFFFBFFF",
        "FFFFFFFBFF",
        "FFFFFHHFBF",
        "FFFFFFFBFF",
        "FFFFFFFFFB",
        "FFFFFFFBFG",  
    ]
        self.is_strawberry_mode_enabled = False
        self.initUI()
    def initUI(self):
        self.setWindowTitle('FrozenLake Training with Controls')
        self.setWindowIcon(QIcon('icon.png'))  
        self.showNormal()  

        # Style CSS pour moderniser l'interface
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
            }
            QGroupBox {
                background-color: #3B4252;
                border: 2px solid #4C566A;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
                color: #ECEFF4;
                font-size: 14px;
            }
            QLabel {
                color: #ECEFF4;
                font-size: 14px;
            }
            QPushButton {
                background-color: #5E81AC;
                color: #ECEFF4;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            QPushButton:pressed {
                background-color: #4C566A;
            }
            QDoubleSpinBox, QSlider {
                background-color: #4C566A;
                color: #ECEFF4;
                border: 1px solid #81A1C1;
                border-radius: 5px;
                padding: 5px;
            }
            QProgressBar {
                background-color: #4C566A;
                color: #ECEFF4;
                border: 1px solid #81A1C1;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #88C0D0;
            }
            QTableWidget {
                background-color: #4C566A;
                color: #ECEFF4;
                border: 1px solid #81A1C1;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #5E81AC;
                color: #ECEFF4;
                padding: 5px;
            }
        """)

        # Widgets
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("background-color: #4C566A; color: #ECEFF4; border-radius: 5px; padding: 10px;")

        self.image_label = QLabel(self)
        self.image_label.setText("Rendu de l'environnement FrozenLake")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("font-size: 16px; color: #ECEFF4;")

        # Barre de progression
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Ajout des labels pour epsilon et les épisodes
        self.epsilon_label = QLabel("Epsilon: 1.00")
        self.episode_label = QLabel("Épisode: 0/1100")
        
        # Style des  labels
        progress_style = """
            QLabel {
                color: #88C0D0;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """
        self.epsilon_label.setStyleSheet(progress_style)
        self.episode_label.setStyleSheet(progress_style)

        # Contrôles pour les paramètres
        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setRange(0.0, 1.0)
        self.learning_rate_spinbox.setValue(0.99)
        self.learning_rate_spinbox.setSingleStep(0.01)
        self.learning_rate_spinbox.valueChanged.connect(self.update_learning_rate)

        self.discount_factor_spinbox = QDoubleSpinBox()
        self.discount_factor_spinbox.setRange(0.0, 1.0)
        self.discount_factor_spinbox.setValue(0.95)
        self.discount_factor_spinbox.setSingleStep(0.01)
        self.discount_factor_spinbox.valueChanged.connect(self.update_discount_factor)

        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 1.0)
        self.epsilon_spinbox.setValue(1.0)
        self.epsilon_spinbox.setSingleStep(0.01)
        self.epsilon_spinbox.valueChanged.connect(self.update_epsilon)

        self.epsilon_decay_spinbox = QDoubleSpinBox()
        self.epsilon_decay_spinbox.setDecimals(3) 
        self.epsilon_decay_spinbox.setRange(0.0, 0.01)  
        self.epsilon_decay_spinbox.setValue(0.001)     
        self.epsilon_decay_spinbox.setSingleStep(0.0001) 
        self.epsilon_decay_spinbox.valueChanged.connect(self.update_epsilon_decay)
        
        self.episodes_spinbox = QDoubleSpinBox()
        self.episodes_spinbox.setRange(1, 100000)
        self.episodes_spinbox.setValue(1100)
        self.episodes_spinbox.setSingleStep(100)
        self.episodes_spinbox.valueChanged.connect(self.update_episodes)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 1000)
        self.speed_slider.setValue(100)

        self.strawberry_mode = QSlider(Qt.Horizontal)
        self.strawberry_mode.setRange(0, 1)  # Deux positions : 0 (off) et 1 (on)
        self.strawberry_mode.setFixedSize(60, 30)
        self.strawberry_mode.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #e0e0e0;
                height: 20px;
                border-radius: 10px;
                margin: 0 5px;
            }
            
            QSlider::sub-page:horizontal:checked {
                background: #34C759;  /* Vert iOS */
                border-radius: 10px;
            }
            
            QSlider::handle:horizontal {
                background: white;
                width: 24px;
                height: 24px;
                border-radius: 12px;
                margin: -2px 0;
            }
            
            QSlider::handle:horizontal:checked {
                background: white;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(Qt.gray)
        shadow.setOffset(2, 2)
        self.strawberry_mode.setGraphicsEffect(shadow)

        # Boutons avec icônes
        self.start_button = QPushButton( "Démarrer l'entraînement")
        self.start_button.clicked.connect(self.start_training)

        self.test_button = QPushButton("Tester l'agent")
        self.test_button.clicked.connect(self.test_agent)
        self.test_button.setEnabled(False)

        self.pause_button = QPushButton( "Pause")
        self.pause_button.clicked.connect(self.toggle_pause)

        self.reset_button = QPushButton( "Reset")
        self.reset_button.clicked.connect(self.reset_application)

        self.save_button = QPushButton( "Sauvegarder le modèle")
        self.save_button.clicked.connect(self.save_model)

        self.load_button = QPushButton( "Charger le modèle")
        self.load_button.clicked.connect(self.load_model)


        self.strawberry_mode_button = QPropertyAnimation(self.strawberry_mode, b"value")
        self.strawberry_mode_button.setDuration(200)
        self.strawberry_mode_button.setEasingCurve(QEasingCurve.OutQuad)
        self.strawberry_mode.valueChanged.connect(self.toggle_strawberry_mode)

        self.strawberry_list = QListWidget()
      
        # Q-Table Display
        self.q_table_widget = QTableWidget(self)
        self.q_table_widget.setRowCount(10)
        self.q_table_widget.setColumnCount(4)
        self.q_table_widget.setHorizontalHeaderLabels(["Left", "Down", "Right", "Up"])
        self.q_table_widget.setVerticalHeaderLabels([f"State {i}" for i in range(10)])
        self.q_table_widget.setEditTriggers(QTableWidget.NoEditTriggers)

        # Créer un widget pour le graphique des récompenses
        self.tab_widget = QTabWidget()
       
        # Onglet Courbe
        self.rewards_canvas = FigureCanvas(plt.Figure())
        self.rewards_ax = self.rewards_canvas.figure.subplots()
        self.rewards_ax.set_xlabel('Épisodes')
        self.rewards_ax.set_ylabel('Récompense')
        self.rewards_ax.set_title('Courbe des récompenses')
        self.rewards_line, = self.rewards_ax.plot([], [], label='Récompense par épisode')
        self.rewards_ax.legend()
        self.tab_widget.addTab(self.rewards_canvas, "Courbe")

         # Onglet Heatmap
        self.heatmap_canvas = FigureCanvas(plt.Figure())
        self.heatmap_ax = self.heatmap_canvas.figure.subplots()
        self.heatmap_img = self.heatmap_ax.imshow(np.zeros((10, 10)), cmap='gray')
        self.heatmap_cbar = self.heatmap_canvas.figure.colorbar(self.heatmap_img, ax=self.heatmap_ax)
        self.heatmap_cbar.set_label('Valeur Q max')
        self.heatmap_ax.set_title('Heatmap (Entraînement)')
        self.tab_widget.addTab(self.heatmap_canvas, "Heatmap")

        self.progress_container = QVBoxLayout()
        self.progress_container.setContentsMargins(0, 0, 0, 0)  # Aucune marge autour du conteneur
        self.progress_container.addWidget(self.progress_bar)
        
        # Layout horizontal pour epsilon_label et episode_label
        self.progress_info_layout = QHBoxLayout()
        self.progress_info_layout.setSpacing(5)  # Espacement réduit entre les labels
        self.progress_info_layout.addWidget(self.epsilon_label)
        self.progress_info_layout.addWidget(self.episode_label) 
        self.progress_container.addLayout(self.progress_info_layout)

        # Layout principal
        main_layout = QHBoxLayout()

        # Layout pour les paramètres et les boutons (gauche)
        left_layout = QVBoxLayout()

        # Groupe pour les paramètres d'entraînement
        training_group = QGroupBox("Paramètres d'Entraînement")
        training_layout = QVBoxLayout()
        training_layout.addWidget(QLabel("Taux d'apprentissage (learning_rate_a):"))
        training_layout.addWidget(self.learning_rate_spinbox)
        training_layout.addWidget(QLabel("Facteur d'actualisation (discount_factor_g):"))
        training_layout.addWidget(self.discount_factor_spinbox)
        training_layout.addWidget(QLabel("Taux d'exploration initial (epsilon):"))
        training_layout.addWidget(self.epsilon_spinbox)
        training_layout.addWidget(QLabel("Taux de décroissance de l'exploration (epsilon_decay_rate):"))
        training_layout.addWidget(self.epsilon_decay_spinbox)
        training_layout.addWidget(QLabel("Nombre d'épisodes (episodes):"))
        training_layout.addWidget(self.episodes_spinbox)
        training_layout.addWidget(QLabel("Vitesse d'entraînement:"))
        training_layout.addWidget(self.speed_slider)
        training_group.setLayout(training_layout)
        training_layout.addWidget(QLabel("🍓 Mode Fraise:"))
        training_layout.addWidget(self.strawberry_mode)
        

        # Groupe pour les boutons de contrôle
        control_buttons_group = QGroupBox("Contrôles")
        control_buttons_layout = QVBoxLayout()
        control_buttons_layout.addWidget(self.start_button)
        control_buttons_layout.addWidget(self.test_button)
        control_buttons_layout.addWidget(self.pause_button)
        control_buttons_layout.addWidget(self.reset_button)
        control_buttons_layout.addWidget(self.save_button)
        control_buttons_layout.addWidget(self.load_button)
        control_buttons_group.setLayout(control_buttons_layout)

        left_layout.addWidget(training_group)
        left_layout.addWidget(control_buttons_group)

        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0) 
        center_layout.addLayout(self.progress_container)
        center_layout.addWidget(self.image_label)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.tab_widget)
        right_layout.addWidget(self.q_table_widget)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Variables pour l'entraînement
        self.episodes = 1100
        self.current_episode = 0
        self.is_training = True
        self.render = True
        self.custom_map =self.custom_map_normal
        self.env = gym.make("FrozenLake-v1", desc=self.custom_map, is_slippery=False, render_mode='rgb_array')
        if self.is_training:
            self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        else:
            # Charger le modèle approprié selon le mode actif
            if self.is_strawberry_mode_enabled:
                try:
                    with open("frozen_lake10x10_strawberry.pkl", "rb") as f:
                        self.q = pickle.load(f)["q_table"]
                except FileNotFoundError:
                    self.text_edit.append("⚠️ Aucun modèle trouvé pour le mode fraise. Réinitialisation de la Q-table.")
                    self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            else:
                try:
                    with open("frozen_lake10x10_normal.pkl", "rb") as f:
                        self.q = pickle.load(f)["q_table"]
                except FileNotFoundError:
                    self.text_edit.append("⚠️ Aucun modèle trouvé pour le mode normal. Réinitialisation de la Q-table.")
                    self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        self.learning_rate_a = 0.95
        self.discount_factor_g = 0.95 
        self.epsilon = 1.0
        self.epsilon_decay_rate =  0.001
        self.rng = np.random.default_rng()
        self.rewards_per_episode = np.zeros(self.episodes) 
        self.update_frame()

        # Timer pour mettre à jour l'interface
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)

        # Timer pour le test
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.run_test_step)

        # Variables pour le test
        self.test_episode = 0
        self.test_state = None
        self.test_terminated = False
        self.test_truncated = False
        self.test_successes = 0
        self.num_test_episodes = 3

    def start_training(self):
        self.reset_environment()
        self.is_training = True
        self.heatmap_ax.set_title('Heatmap des Q-values (Entraînement)')
        self.timer.start(100)

    def update_learning_rate(self):
        self.learning_rate_a = self.learning_rate_spinbox.value()

    def update_discount_factor(self):
        self.discount_factor_g = self.discount_factor_spinbox.value()

    def update_epsilon(self):
        self.epsilon = self.epsilon_spinbox.value()

    def update_epsilon_decay(self):
        self.epsilon_decay_rate = self.epsilon_decay_spinbox.value()

    def update_episodes(self):
        self.episodes = int(self.episodes_spinbox.value())
        self.rewards_per_episode = np.zeros(self.episodes)

 

    def update_ui(self):
        if self.current_episode < self.episodes:
            self.run_episode()
            self.current_episode += 1
            progress = int((self.current_episode / self.episodes) * 100)
            self.progress_bar.setValue(progress)
            self.update_frame()
            self.update_rewards_plot()
            self.update_heatmap()
            self.update_q_table()
            self.epsilon_label.setText(f"Epsilon: {self.epsilon:.4f}")
            self.episode_label.setText(f"Épisode: {self.current_episode}/{self.episodes}")
        else:
            self.timer.stop()
            self.env.close()
            self.text_edit.append("Entraînement terminé !")
            self.test_button.setEnabled(True)
            if self.is_training:
                if self.is_strawberry_mode_enabled:
                    filename = "frozen_lake10x10_strawberry.pkl"
                else:
                    filename = "frozen_lake10x10_normal.pkl"

                # Sauvegarder la Q-table ET les récompenses
                data = {
                    "q_table": self.q,
                    "rewards": self.rewards_per_episode
                }
                with open(filename, "wb") as f:
                    pickle.dump(data, f)

                self.text_edit.append(f"Modèle sauvegardé avec succès à : {filename}")

            self.is_training = False
            self.heatmap_ax.set_title('Heatmap des Q-values (Test)')
            self.update_heatmap()

    
    def toggle_strawberry_mode(self):
        try:
            # Vérifie la valeur actuelle du slider (0 = désactivé, 1 = activé)
            if self.strawberry_mode.value() == 1:
                self.is_strawberry_mode_enabled = True
                self.text_edit.append("🍓 Mode fraise activé ! Grille mise à jour.")
                self.custom_map = self.custom_map_strawberry  # Utiliser la grille avec les fraises
                
                # Afficher les positions des fraises dans la liste
                self.strawberry_list.clear()  # Nettoyer la liste avant de l'actualiser
                for row_idx, row in enumerate(self.custom_map):
                    for col_idx, cell in enumerate(row):
                        if cell == 'B':  # 'B' représente une fraise
                            self.strawberry_list.addItem(f"Fraise à ({row_idx}, {col_idx})")
                
                if hasattr(self, "strawberry_mode_button"):
                    self.strawberry_mode_button.setText("Désactiver le mode fraise")
            
            else:
                self.is_strawberry_mode_enabled = False
                self.text_edit.append("❌ Mode fraise désactivé ! Grille réinitialisée.")
                self.custom_map = self.custom_map_normal  # Revenir à la grille normale
                self.strawberry_list.clear()  # Vider la liste des fraises
                
                if hasattr(self, "strawberry_mode_button"):
                    self.strawberry_mode_button.setText("Activer le mode fraise")

                if self.strawberry_mode.value() == 0:
                    # Réinitialiser l'environnement avec la nouvelle grille
                    self.reset_application()
                    # Réinitialiser la Q-table et les variables d'apprentissage
                    self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n)) 
                    self.current_episode = 0 
                    self.rewards_per_episode = np.zeros(self.episodes)  
                    self.collected_strawberries = set() 
                    
                    self.update_frame()
                    self.update_rewards_plot()
                    self.update_heatmap()
                    self.update_q_table()

                # Mettre à jour la liste des fraises dans l'interface
                self.strawberry_list.clear()  
                if self.is_strawberry_mode_enabled:
                    for row_idx, row in enumerate(self.custom_map):
                        for col_idx, cell in enumerate(row):
                            if cell == 'B':  
                                self.strawberry_list.addItem(f"Fraise à ({row_idx}, {col_idx})")

        except AttributeError as e:
             print(f"Erreur : {e}. Vérifiez que self.strawberry_mode est un QSlider.")
                
               
    def run_episode(self):
        total_reward = 0
        state = self.env.reset()[0]
        terminated = False
        truncated = False
        strawberry_collected = 0  

        while not terminated and not truncated:
            if self.is_training and self.rng.random() < self.epsilon:
                action = self.env.action_space.sample()  # Exploration
            else:
                action = np.argmax(self.q[state, :])  # Exploitation

            new_state, _, terminated, truncated, _ = self.env.step(action)
            row = new_state // len(self.custom_map)
            col = new_state % len(self.custom_map[0])

            # Calcul de la récompense basée sur la position actuelle
            if self.custom_map[row][col] == 'H':
                reward = -10  # Pénalité forte pour les trous
            elif self.custom_map[row][col] == 'G':
                if self.is_strawberry_mode_enabled and strawberry_collected < 3:
                    reward = -10  # Pénalité si G est atteint sans avoir collecté assez de fraises
                else:
                    reward = 100  # Récompense maximale pour atteindre G
            elif self.custom_map[row][col] == 'B' and (row, col) not in self.collected_strawberries:
                if self.is_strawberry_mode_enabled:
                    reward = 50  # Récompense élevée pour collecter une fraise en mode fraise
                    strawberry_collected += 1
                    self.collected_strawberries.add((row, col))
                else:
                    reward = 0  # Pas de récompense pour les fraises en mode normal
            else:
                distance = abs(9 - row) + abs(9 - col)
                reward = 0.1 * (1 / (distance + 1))  # Récompense basée sur la distance vers G

            # Mise à jour Q-learning
            if self.is_training:
                self.q[state, action] += self.learning_rate_a * (
                    reward + self.discount_factor_g * np.max(self.q[new_state, :]) - self.q[state, action]
                )

            state = new_state
            total_reward += reward

        # Stocker la récompense totale pour cet épisode
        self.rewards_per_episode[self.current_episode] = total_reward
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
        self.collected_strawberries.clear()  # Réinitialiser les fraises collectées pour le prochain épisode

    def update_frame(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                frame = self.env.render()
                if frame is not None:
                    img = Image.fromarray(frame)
                    img = img.convert("RGB")
                    
                    # Draw strawberries if in strawberry mode
                    if self.strawberry_mode:
                        try:
                            strawberry_img = Image.open('strawberry.png').convert("RGBA")
                            strawberry_img = ImageOps.fit(strawberry_img, (img.width//10, img.height//10), Image.Resampling.LANCZOS)
                            for row in range(len(self.custom_map)):
                                for col in range(len(self.custom_map[row])):
                                    if self.custom_map[row][col] == 'B':
                                        x = col * (img.width // 10)
                                        y = row * (img.height // 10)
                                        img.paste(strawberry_img, (x, y), strawberry_img)
                        except Exception as e:
                            print(f"Error drawing strawberries: {e}")

                    data = img.tobytes("raw", "RGB")
                    qimage = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
                    self.image_label.setPixmap(QPixmap.fromImage(qimage))
            except gym.error.ResetNeeded:
                self.text_edit.append("Erreur : L'environnement doit être réinitialisé avant le rendu.")
                self.reset_environment()
    def update_rewards_plot(self):
        self.rewards_line.set_data(range(len(self.rewards_per_episode)), self.rewards_per_episode)
        self.rewards_ax.relim()
        self.rewards_ax.autoscale_view()
        self.rewards_canvas.draw()

    def update_heatmap(self):
        q_max = np.max(self.q, axis=1)
        q_grid = q_max.reshape((10, 10))
        
        # Calculer le minimum et maximum des valeurs Q
        q_min = np.min(q_grid)  
        q_max_value = np.max(q_grid)
        
        if self.is_training:
            self.heatmap_img.set_cmap('gray')
            self.heatmap_ax.set_title('Heatmap (Entraînement)')
        else:
            self.heatmap_img.set_cmap('viridis')
            self.heatmap_ax.set_title('Heatmap (Test)')
            
        self.heatmap_img.set_data(q_grid)
        self.heatmap_img.set_clim(vmin=q_min, vmax=q_max_value)  
        self.heatmap_canvas.draw()


    def update_q_table(self):
        for state in range(self.q.shape[0]):
            for action in range(self.q.shape[1]):
                item = QTableWidgetItem(f"{self.q[state, action]:.2f}")
                if action == np.argmax(self.q[state, :]):
                    item.setBackground(Qt.green)
                self.q_table_widget.setItem(state, action, item)

    def test_agent(self):
        self.reset_environment()
        self.collected_strawberries = set()
        path = []
        self.text_edit.append("\nDébut du test AUTOMATIQUE...")
        self.text_edit.append("\nDébut du test de l'agent...")
        self.test_episode = 0
        self.test_successes = 0
        self.test_state = self.env.reset()[0]
        self.test_terminated = False
        self.test_truncated = False
        self.test_timer.start(500)

        while not self.test_terminated and not self.test_truncated:
            action = np.argmax(self.q[self.test_state, :])  # Choix de l'action basé sur la Q-table
            new_state, _, self.test_terminated, self.test_truncated, _ = self.env.step(action)

            # Ajouter l'état courant au chemin
            row = new_state // len(self.custom_map)
            col = new_state % len(self.custom_map[0])
            path.append((row, col))

            # Vérifier si l'agent a collecté une fraise
            if self.custom_map[row][col] == 'B' and (row, col) not in self.collected_strawberries:
                self.collected_strawberries.add((row, col))
                self.text_edit.append(f"🍓 Fraise collectée à ({row}, {col})")

            # Vérifier si l'agent a atteint l'état terminal
            if self.custom_map[row][col] == 'G':
                if self.is_strawberry_mode_enabled:
                    if len(self.collected_strawberries) >= 3:
                        self.test_successes += 1
                        self.text_edit.append("✅ Objectif atteint avec suffisamment de fraises collectées !")
                    else:
                        self.text_edit.append("❌ Objectif atteint, mais pas assez de fraises collectées.")
                else:
                    self.test_successes += 1
                    self.text_edit.append("✅ Objectif atteint !")

            self.test_state = new_state
            self.update_frame()

        # Afficher le chemin suivi par l'agent
        print(f"Chemin suivi : {path}")
        self.visualize_path(path)
        self.text_edit.append(f"\nRésultat du test : {self.test_successes} succès.")
    
    def visualize_path(self, path):
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((10, 10)), cmap='Blues')
        
        # Marqueurs spéciaux
        for (row, col) in path:
            if self.custom_map[row][col] == 'B':
                ax.scatter(col, row, marker='*', s=200, color='red')  # Fraises
            elif self.custom_map[row][col] == 'G':
                ax.scatter(col, row, marker='D', s=200, color='gold')  # But
            else:
                ax.scatter(col, row, marker='o', s=50, color='lime')  # Chemin
        
        ax.set_title("Chemin suivi par l'agent")
        plt.show()
        
        # Marqueurs spéciaux
        for (row, col) in path:
            if self.custom_map[row][col] == 'B':
                ax.scatter(col, row, marker='*', s=200, color='red')  # Fraises
            else:
                ax.scatter(col, row, marker='o', s=50, color='lime')  # Chemin
        
        ax.set_title("Chemin optimal de l'agent")
        plt.show()

    def run_test_step(self):
        if self.test_episode < self.num_test_episodes:
            if not self.test_terminated and not self.test_truncated:
                action = np.argmax(self.q[self.test_state, :])
                new_state, reward, self.test_terminated, self.test_truncated, _ = self.env.step(action)

                # Récompense supplémentaire en mode fraise
                if self.strawberry_mode and self.custom_map[new_state // len(self.custom_map)][new_state % len(self.custom_map[0])] == 'B':
                    reward += 15  

                if self.custom_map[new_state // len(self.custom_map)][new_state % len(self.custom_map[0])] == 'G':
                    reward = 50
                    self.test_successes += 1

                self.test_state = new_state
                self.update_frame()
                self.text_edit.append(f"Test Épisode {self.test_episode + 1} - Étape - Récompense : {reward}")

            else:
                self.test_episode += 1
                if self.test_episode < self.num_test_episodes:
                    self.test_state = self.env.reset()[0]
                    self.test_terminated = False
                    self.test_truncated = False
                else:
                    self.test_timer.stop()
                    self.text_edit.append(f"\nRésultat du test : {self.test_successes} succès sur {self.num_test_episodes} épisodes.")
    
    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_button.setText("Reprendre")
        else:
            self.timer.start(100)
            self.pause_button.setText("Pause")
    
    def reset_parameters(self):
        self.learning_rate_a = 0.9
        self.discount_factor_g = 0.95
        self.epsilon = 1.0  
        self.epsilon_decay_rate = 0.01
        self.episodes = 1100  

        self.learning_rate_spinbox.setValue(self.learning_rate_a)
        self.discount_factor_spinbox.setValue(self.discount_factor_g)
        self.epsilon_spinbox.setValue(self.epsilon)
        self.epsilon_decay_spinbox.setValue(self.epsilon_decay_rate)
        self.episodes_spinbox.setValue(self.episodes)

        self.text_edit.append("Paramètres réinitialisés avec succès !")
        self.text_edit.append("Paramètres réinitialisés avec succès !")
        
    def save_model(self):
        base_filename, _ = QFileDialog.getSaveFileName(
            self,
            "Sauvegarder le modèle",
            "",
            "Fichiers Pickle (*.pkl)"
        )
        if base_filename:
            # Ajouter un suffixe selon le mode actif
            filename = f"{base_filename}_strawberry.pkl" if self.is_strawberry_mode_enabled else f"{base_filename}_normal.pkl"
        
            try:
                # Sauvegarder toutes les données pertinentes dans un dictionnaire
                data = {
                    "q_table": self.q,
                    "rewards_per_episode": self.rewards_per_episode,
                    "is_strawberry_mode_enabled": self.is_strawberry_mode_enabled,
                    "episodes": self.episodes,
                    "learning_rate_a": self.learning_rate_a,
                    "discount_factor_g": self.discount_factor_g,
                    "epsilon": self.epsilon,
                    "epsilon_decay_rate": self.epsilon_decay_rate
                }
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
                self.text_edit.append(f"Modèle et courbe sauvegardés avec succès à : {filename}")
                
                # Boîte de dialogue personnalisée pour la sauvegarde réussie
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Sauvegarde réussie")
                msg_box.setText("Modèle sauvegardé avec succès!")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #2E3440;
                    }
                    QMessageBox QLabel {
                        color: #ECEFF4;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton {
                        background-color: #5E81AC;
                        color: #ECEFF4;
                        border: none;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton:hover {
                        background-color: #81A1C1;
                    }
                    QMessageBox QPushButton:pressed {
                        background-color: #4C566A;
                    }
                """)
                msg_box.exec_()
            
            except Exception as e:
                self.text_edit.append(f"Erreur de sauvegarde : {str(e)}")
                # Boîte de dialogue d'erreur personnalisée
                error_box = QMessageBox(self)
                error_box.setWindowTitle("Erreur")
                error_box.setText(f"Échec de la sauvegarde: {str(e)}")
                error_box.setIcon(QMessageBox.Critical)
                error_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #2E3440;
                    }
                    QMessageBox QLabel {
                        color: #ECEFF4;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton {
                        background-color: #5E81AC;
                        color: #ECEFF4;
                        border: none;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton:hover {
                        background-color: #81A1C1;
                    }
                    QMessageBox QPushButton:pressed {
                        background-color: #4C566A;
                    }
                """)
                error_box.exec_()

    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Charger un modèle",
            "",
            "Fichiers Pickle (*.pkl)"
        )
        if filename:
            try:
                # Charger toutes les données depuis le fichier
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                
                # Restaurer les variables sauvegardées
                self.q = data.get("q_table", np.zeros((self.env.observation_space.n, self.env.action_space.n)))
                self.rewards_per_episode = data.get("rewards_per_episode", np.zeros(self.episodes))
                self.is_strawberry_mode_enabled = data.get("is_strawberry_mode_enabled", False)
                self.episodes = data.get("episodes", 1100)
                self.learning_rate_a = data.get("learning_rate_a", 0.95)
                self.discount_factor_g = data.get("discount_factor_g", 0.95)
                self.epsilon = data.get("epsilon", 1.0)
                self.epsilon_decay_rate = data.get("epsilon_decay_rate", 0.001)
                
                # Réinitialiser l'environnement avec la grille correspondante
                self.custom_map = self.custom_map_strawberry if self.is_strawberry_mode_enabled else self.custom_map_normal
                self.env.close()
                self.env = gym.make("FrozenLake-v1", desc=self.custom_map, is_slippery=False, render_mode='rgb_array')
                
                # Mettre à jour les spinboxes avec les valeurs restaurées
                self.learning_rate_spinbox.setValue(self.learning_rate_a)
                self.discount_factor_spinbox.setValue(self.discount_factor_g)
                self.epsilon_spinbox.setValue(self.epsilon)
                self.epsilon_decay_spinbox.setValue(self.epsilon_decay_rate)
                self.episodes_spinbox.setValue(self.episodes)
                
                # Mettre à jour l'affichage
                self.start_button.setEnabled(False)
                self.pause_button.setEnabled(False)
                self.test_button.setEnabled(True)
                self.update_q_table()
                self.update_rewards_plot()
                self.heatmap_ax.set_title('Heatmap des Q-values (Test)')
                self.update_heatmap()
                self.progress_bar.setValue(0)
                self.test_state = self.env.reset()[0]
                self.update_frame()
                
                self.text_edit.append(f"Modèle chargé depuis : {filename}")
                
                # Boîte de dialogue personnalisée pour le chargement réussi
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Chargement réussi")
                msg_box.setText("Modèle chargé avec succès!")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #2E3440;
                    }
                    QMessageBox QLabel {
                        color: #ECEFF4;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton {
                        background-color: #5E81AC;
                        color: #ECEFF4;
                        border: none;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton:hover {
                        background-color: #81A1C1;
                    }
                    QMessageBox QPushButton:pressed {
                        background-color: #4C566A;
                    }
                """)
                msg_box.exec_()
            
            except Exception as e:
                self.text_edit.append(f"Erreur lors du chargement : {str(e)}")
                # Boîte de dialogue d'erreur personnalisée
                error_box = QMessageBox(self)
                error_box.setWindowTitle("Erreur")
                error_box.setText(f"Échec du chargement: {str(e)}")
                error_box.setIcon(QMessageBox.Critical)
                error_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #2E3440;
                    }
                    QMessageBox QLabel {
                        color: #ECEFF4;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton {
                        background-color: #5E81AC;
                        color: #ECEFF4;
                        border: none;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 14px;
                    }
                    QMessageBox QPushButton:hover {
                        background-color: #81A1C1;
                    }
                    QMessageBox QPushButton:pressed {
                        background-color: #4C566A;
                    }
                """)
                error_box.exec_()
                    
    def reset_application(self):
        """
        Réinitialise complètement l'application, y compris les paramètres utilisateur et les boutons.
        """
        # Réinitialiser l'environnement avec la grille actuelle
        self.reset_environment()

        # Réinitialiser la Q-table
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        # Réinitialiser les variables d'apprentissage
        self.current_episode = 0
        self.rewards_per_episode = np.zeros(self.episodes)
        self.collected_strawberries = set()

        # Réinitialiser les variables de test
        self.test_episode = 0
        self.test_successes = 0
        self.test_state = self.env.reset()[0]
        self.test_terminated = False
        self.test_truncated = False

        # Réinitialiser les paramètres d'apprentissage aux valeurs par défaut
        self.learning_rate_a = 0.95
        self.discount_factor_g = 0.95 
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.001
        self.episodes = 1100

        # Mettre à jour les spinboxes avec les valeurs par défaut
        self.learning_rate_spinbox.setValue(self.learning_rate_a)
        self.discount_factor_spinbox.setValue(self.discount_factor_g)
        self.epsilon_spinbox.setValue(self.epsilon)
        self.epsilon_decay_spinbox.setValue(self.epsilon_decay_rate)
        self.episodes_spinbox.setValue(self.episodes)

        # Réinitialiser les boutons
        self.start_button.setEnabled(True)  
        self.pause_button.setEnabled(True)  
        self.test_button.setEnabled(False)  
        self.pause_button.setText("Pause")  
        self.timer.stop()  

        # Réinitialiser le mode fraise
        self.is_strawberry_mode_enabled = False
        self.strawberry_mode.setValue(0) 
        self.custom_map = self.custom_map_normal  
        self.strawberry_list.clear()  
        # Réinitialiser l'interface
        self.progress_bar.setValue(0)  
        self.text_edit.append("✅ Application réinitialisée avec succès.")  

        if hasattr(self, "epsilon_label"):
            self.epsilon_label.setText(f"Epsilon: {self.epsilon:.4f}")
        if hasattr(self, "episode_label"):
            self.episode_label.setText(f"Épisode: {self.current_episode}/{self.episodes}")

        # Mettre à jour les graphiques
        self.update_rewards_plot()
        self.update_heatmap()
        self.update_q_table()

        # Mettre à jour l'affichage de l'environnement
        self.update_frame()

    def reset_environment(self):
        """Réinitialise l'environnement avec la grille actuelle"""
        self.env.close()
        self.env = gym.make("FrozenLake-v1", desc=self.custom_map, is_slippery=False, render_mode='rgb_array')
        self.test_state = self.env.reset()[0]  
        self.update_frame()  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FrozenLakeApp()
    window.show()
    sys.exit(app.exec_())