from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects

class PersonRecognitionSystem:
    def __init__(self):
        self.reset_model()
        
    def reset_model(self):
        """Resetuje model do stanu początkowego"""
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.current_person_embedding = None
        
    def extract_frames(self, video_path, frame_interval=30):
        """Ekstrahuje klatki z filmu"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            frame_count += 1
            
        cap.release()
        return frames

    def get_embedding(self, image):
        """Generuje embedding dla pojedynczego obrazu"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        faces = self.mtcnn(image)
        if faces is not None:
            if not isinstance(faces, torch.Tensor):
                return None
            if faces.ndim == 3:
                faces = faces.unsqueeze(0)
            embeddings = self.resnet(faces)
            return embeddings.detach()
        return None

    def train_on_person(self, video_path):
        """Uczy się rozpoznawać osobę z jej filmu treningowego"""
        print(f"\nUczenie się rozpoznawania z filmu: {video_path}")
        
        embeddings = []
        frames = self.extract_frames(video_path)
        
        for frame in tqdm(frames, desc="Przetwarzanie klatek treningowych"):
            emb = self.get_embedding(frame)
            if emb is not None:
                embeddings.append(emb[0])
                
        if embeddings:
            self.current_person_embedding = torch.stack(embeddings).mean(dim=0)
            return True
        return False

    def compare_with_video(self, video_path, threshold=0.7):
        """Porównuje nauczoną osobę z filmem testowym"""
        if self.current_person_embedding is None:
            return 0.0
        
        detections = []
        frames = self.extract_frames(video_path)
        
        for frame in tqdm(frames, desc="Porównywanie"):
            emb = self.get_embedding(frame)
            if emb is not None:
                similarity = torch.nn.functional.cosine_similarity(
                    self.current_person_embedding.unsqueeze(0),
                    emb[0].unsqueeze(0)
                ).item()
                detections.append(similarity)
                
        if detections:
            return np.mean(detections)
        return 0.0

def wykonaj_test1(base_path):
    """Test 1: Porównanie wszystkich osób ze wszystkimi"""
    system = PersonRecognitionSystem()
    persons = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    results_matrix = np.zeros((len(persons), len(persons)))
    
    for i, train_person in enumerate(persons):
        system.reset_model()
        train_video = os.path.join(base_path, train_person, os.listdir(os.path.join(base_path, train_person))[0])
        if system.train_on_person(train_video):
            for j, test_person in enumerate(persons):
                test_video = os.path.join(base_path, test_person, os.listdir(os.path.join(base_path, test_person))[0])
                similarity = system.compare_with_video(test_video)
                results_matrix[i, j] = similarity
    
    zapisz_wyniki(results_matrix, persons, "test1")
    return results_matrix

def wykonaj_test2(base_path):
    """Test 2: Porównanie dwóch ujęć każdej osoby"""
    system = PersonRecognitionSystem()
    persons = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    total_videos = len(persons) * 2
    results_matrix = np.zeros((total_videos, total_videos))
    
    labels = []
    video_paths = []
    
    for person in persons:
        person_dir = os.path.join(base_path, person)
        videos = [os.path.join(person_dir, v) for v in os.listdir(person_dir)][:2]
        video_paths.extend(videos)
        labels.extend([f"{person}_1", f"{person}_2"])
    
    for i, train_video in enumerate(video_paths):
        system.reset_model()
        if system.train_on_person(train_video):
            for j, test_video in enumerate(video_paths):
                similarity = system.compare_with_video(test_video)
                results_matrix[i, j] = similarity
    
    zapisz_wyniki(results_matrix, labels, "test2")
    return results_matrix

def wykonaj_test3(base_path):
    """Test 3: Porównanie ujęć o różnych długościach - każde z każdym"""
    system = PersonRecognitionSystem()
    persons = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    czasy = [5, 10, 15]
    
    print("\nSprawdzanie struktury katalogów:")
    print(f"Katalog bazowy: {base_path}")
    print(f"Znalezione katalogi osób: {persons}")
    
    # Zbieranie wszystkich nagrań
    all_videos = []  # Lista krotek (osoba, ścieżka, ujęcie, czas)
    labels = []      # Lista etykiet dla macierzy
    
    for person in persons:
        person_dir = os.path.join(base_path, person)
        print(f"\nSprawdzanie katalogu: {person}")
        files = sorted(os.listdir(person_dir))
        print(f"Znalezione pliki: {files}")
        
        for file in files:
            video_path = os.path.join(person_dir, file)
            # Dodaj plik do listy bez względu na jego nazwę
            all_videos.append((person, video_path))
            labels.append(f"{person}_{file.replace('.mp4', '')}")
    
    total_videos = len(all_videos)
    print(f"\nZnalezione pliki wideo: {total_videos}")
    print(f"Liczba osób: {len(persons)}")
    print("\nZnalezione nagrania:")
    for label in labels:
        print(f"- {label}")
    
    # Tworzymy macierz 24x24
    results_matrix = np.zeros((total_videos, total_videos))
    
    # Porównywanie każdego nagrania z każdym
    for i, (train_person, train_video) in enumerate(tqdm(all_videos, desc="Przetwarzanie nagrań")):
        system.reset_model()
        print(f"\nUczenie na: {train_person} - {os.path.basename(train_video)}")
        
        if system.train_on_person(train_video):
            for j, (test_person, test_video) in enumerate(all_videos):
                similarity = system.compare_with_video(test_video)
                results_matrix[i, j] = similarity
                print(f"Podobieństwo do {test_person} ({os.path.basename(test_video)}): {similarity:.2f}")
    
    print("\nWymiary macierzy wynikowej:", results_matrix.shape)
    print("Liczba etykiet:", len(labels))
    
    zapisz_wyniki(results_matrix, labels, labels, "test3")
    return results_matrix

def generuj_elegancka_macierz(matrix, labels, title, output_path):
    """Generuje elegancką macierz z siatką i lepszym formatowaniem"""
    plt.figure(figsize=(15, 12))
    
    # Tworzenie własnej mapy kolorów
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#3689e6', 'white', '#c6262e']  # niebieski -> biały -> czerwony
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # Ustawienia głównej macierzy
    im = plt.imshow(matrix, cmap=custom_cmap, aspect='equal')
    
    # Usunięcie domyślnej siatki
    plt.grid(False)
    
    # Dodanie tylko obramowania komórek
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    
    # Konfiguracja kolorowej skali
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Podobieństwo', rotation=270, labelpad=15)
    
    # Konfiguracja etykiet - na górze
    plt.xticks(range(len(labels)), labels, rotation=45, ha='left', va='bottom')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    plt.yticks(range(len(labels)), labels)
    
    # Dodanie wartości do komórek
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            # Dostosowanie koloru tekstu
            color = 'white' if 0.3 < value < 0.7 else 'black'
            # Dodanie wartości z czarnym obramowaniem tekstu dla lepszej czytelności
            plt.text(j, i, f'{value:.2f}', 
                    ha='center', va='center', 
                    color=color,
                    fontsize=9,
                    path_effects=[
                        path_effects.withStroke(linewidth=2, foreground='black'),
                        path_effects.Normal()
                    ])
    
    # Tytuł i etykiety osi
    plt.title(title, pad=20, fontsize=14, fontweight='bold', y=1.15)
    plt.xlabel('Osoba testowa', labelpad=10)
    plt.ylabel('Osoba wzorcowa', labelpad=10)
    
    # Dostosowanie układu
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Zapisanie wykresu
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def zapisz_wyniki(matrix, row_labels, col_labels, test_name):
    """Zapisuje wyniki testu do plik��w CSV i PNG"""
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Zapis do CSV
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.to_csv(f'results/wyniki_{test_name}.csv')
    
    # Wizualizacja macierzy
    plt.figure(figsize=(15, 12))
    
    # Tworzenie własnej mapy kolorów
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#3689e6', 'white', '#c6262e']  # niebieski -> biały -> czerwony
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # Ustawienia głównej macierzy
    im = plt.imshow(matrix, cmap=custom_cmap, aspect='equal')
    
    # Usunięcie domyślnej siatki
    plt.grid(False)
    
    # Dodanie tylko obramowania komórek
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    
    # Konfiguracja kolorowej skali
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Podobieństwo', rotation=270, labelpad=15)
    
    # Konfiguracja etykiet
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha='left', va='bottom')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    plt.yticks(range(len(row_labels)), row_labels)
    
    # Dodanie wartości do komórek
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            # Dostosowanie koloru tekstu
            color = 'white' if 0.3 < value < 0.7 else 'black'
            # Dodanie wartości z czarnym obramowaniem tekstu
            plt.text(j, i, f'{value:.2f}', 
                    ha='center', va='center', 
                    color=color,
                    fontsize=9,
                    path_effects=[
                        path_effects.withStroke(linewidth=2, foreground='black'),
                        path_effects.Normal()
                    ])
    
    # Tytuł i etykiety osi
    plt.title(f'Macierz porównań - {test_name}', pad=20, fontsize=14, fontweight='bold', y=1.15)
    plt.xlabel('Osoba testowa', labelpad=10)
    plt.ylabel('Osoba wzorcowa', labelpad=10)
    
    # Dostosowanie układu
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Zapisanie wykresu
    plt.savefig(f'results/macierz_{test_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def generuj_macierze_z_csv():
    """Generuje macierze na podstawie istniejących plików CSV"""
    if not os.path.exists('results'):
        print("Folder 'results' nie istnieje!")
        return
    
    csv_files = [f for f in os.listdir('results') if f.endswith('.csv') and f.startswith('wyniki_')]
    
    for csv_file in csv_files:
        test_name = csv_file.replace('wyniki_', '').replace('.csv', '')
        print(f"Generuję macierz dla {test_name}...")
        
        # Wczytaj dane z CSV
        df = pd.read_csv(f'results/{csv_file}', index_col=0)
        
        # Generuj elegancką macierz
        generuj_elegancka_macierz(
            matrix=df.values,
            labels=df.columns,
            title=f'Macierz porównań - {test_name}',
            output_path=f'results/macierz_{test_name}.png'
        )
        
        print(f"Zapisano macierz do results/macierz_{test_name}.png")

def wybierz_tryb():
    """Funkcja do wyboru trybu działania programu"""
    print("\n=== SYSTEM ROZPOZNAWANIA OSÓB ===")
    print("1. Generuj macierze z istniejących plików CSV")
    print("2. Wykonaj testy")
    
    while True:
        try:
            wybor = int(input("\nWybierz tryb (1-2): "))
            if wybor in [1, 2]:
                return wybor
            print("Nieprawidłowy wybór. Wybierz 1 lub 2.")
        except ValueError:
            print("Nieprawidłowy wybór. Wprowadź liczbę.")

def wybierz_testy():
    """Funkcja do wyboru testów do wykonania"""
    print("\n=== WYBÓR TESTÓW ===")
    print("1. Wszystkie testy")
    print("2. Test 1 (porównanie wszystkich osób)")
    print("3. Test 2 (porównanie dwóch ujęć)")
    print("4. Test 3 (porównanie różnych długości)")
    
    while True:
        try:
            wybor = int(input("\nWybierz opcję (1-4): "))
            if wybor == 1:
                return ['test1', 'test2', 'test3']
            elif wybor in [2, 3, 4]:
                return [f'test{wybor-1}']
            print("Nieprawidłowy wybór. Wybierz 1-4.")
        except ValueError:
            print("Nieprawidłowy wybór. Wprowadź liczbę.")

def main():
    base_paths = {
        'test1': './dataset/test1',
        'test2': './dataset/test2',
        'test3': './dataset/test3'
    }
    
    tryb = wybierz_tryb()
    
    if tryb == 1:
        print("\nGenerowanie macierzy z istniejących plików CSV...")
        generuj_macierze_z_csv()
    else:
        testy_do_wykonania = wybierz_testy()
        print(f"\nWybrane testy: {', '.join(testy_do_wykonania)}")
        
        for test in testy_do_wykonania:
            print(f"\nRozpoczynam {test.upper()}...")
            if test == 'test1':
                wykonaj_test1(base_paths[test])
            elif test == 'test2':
                wykonaj_test2(base_paths[test])
            elif test == 'test3':
                wykonaj_test3(base_paths[test])
            print(f"Zakończono {test.upper()}")

if __name__ == "__main__":
    main()