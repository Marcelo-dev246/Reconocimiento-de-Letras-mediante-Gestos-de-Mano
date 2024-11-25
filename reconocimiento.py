import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import time  # Importar módulo de tiempo

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

# Inicialización de Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# Cambiar el índice según la cámara que desees usar (0 para la primera, 1 para la segunda, etc.)
cap = cv2.VideoCapture(1)  # Usar la cámara web

# Puntos de referencia para cada dedo
thumb_points = [1, 2, 4]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Colores
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

# Mapear configuraciones de dedos a letras
hand_signs = {
    (False, False, False, False, False): ' ',  # Espacio
    (True, False, False, False, False): 'A',
    (False, True, False, False, False): 'B',
    (True, False, True, True, True): 'C',
    (False, False, False, True, False): 'D',
    (False, False, False, False, True): 'E',
    (True, True, False, False, False): 'F',
    (True, False, True, False, False): 'G',
    (True, True, True, True, True): 'H',
    (True, False, False, False, True): 'I',
    (False, True, True, False, False): 'J',
    (False, True, False, True, False): 'K',
    (False, True, False, False, True): 'L',
    (False, False, True, True, False): 'M',
    (False, False, True, False, True): 'N',
    (False, False, False, True, True): 'O',
    (True, True, True, False, False): 'P',
    (True, True, False, True, False): 'Q',
    (True, True, False, False, True): 'R',
    (True, False, True, True, False): 'S',
    (True, False, True, False, True): 'T',
    (True, False, False, True, True): 'U',
    (False, True, True, True, False): 'V',
    (False, True, True, False, True): 'W',
    (False, True, False, True, True): 'X',
    (False, False, True, True, True): 'Y',
    (True, True, True, True, False): 'Z',
}

# Estado del cuadro de texto
writing_mode = False
text_output = ""
last_detected_time = time.time()  # Tiempo inicial para el temporizador

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        fingers_counter = "_"
        thickness = [2, 2, 2, 2, 2]
        
        if results.multi_hand_landmarks:
            coordinates_thumb = []
            coordinates_palm = []
            coordinates_ft = []
            coordinates_fb = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer coordenadas de puntos clave
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])
                
                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])
                
                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])
                
                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])
                
                # Calcular si el pulgar está extendido
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                thumb_finger = angle > 150
                
                # Calcular si los demás dedos están extendidos
                nx, ny = palm_centroid(coordinates_palm)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)
                d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                dif = d_centrid_ft - d_centrid_fb
                fingers = dif > 0
                fingers = np.append(thumb_finger, fingers)
                fingers_counter = str(np.count_nonzero(fingers))
                
                # Determinar letra con temporizador
                current_time = time.time()
                if writing_mode and (current_time - last_detected_time > 3):  # 3 segundos
                    hand_tuple = tuple(fingers)
                    detected_char = hand_signs.get(hand_tuple, "")
                    text_output += detected_char
                    last_detected_time = current_time  # Actualizar tiempo

                for (i, finger) in enumerate(fingers):
                    if finger:
                        thickness[i] = -1
                
                # Dibujar conexiones y landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Visualización
        cv2.rectangle(frame, (0, 0), (80, 80), (125, 220, 0), -1)
        cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)
        
        # Cuadro de texto
        cv2.rectangle(frame, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.putText(frame, text_output, (20, height - 20), 1, 2, (255, 255, 255), 2)
        cv2.putText(frame, "Presione u para escribir con senas, t para nueva oracion", 
                    (10, height - 100), 1, 1, (255, 255, 255), 1)
        
        cv2.imshow("Frame", frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('u'):
            writing_mode = not writing_mode
        elif key == ord('t'):
            text_output = ""
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()