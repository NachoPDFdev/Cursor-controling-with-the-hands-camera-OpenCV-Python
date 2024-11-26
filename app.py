"""
Hand Gesture Mouse Control
Author: NachoPDFdev (https://github.com/NachoPDFdev)

This application uses computer vision and hand gesture recognition to control
the mouse cursor. It captures hand movements through the webcam and translates
different hand gestures into mouse actions:
- Open palm: Move cursor
- Index finger up: Prepare for click
- Closed fist: Click
- Move and click gesture: Drag and click
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from datetime import datetime
import random
import pyautogui
import time

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Directory to store face images
face_dir = 'faces'
if not os.path.exists(face_dir):
    os.makedirs(face_dir)

# Directory for hand images
hand_dir = 'hands'
hand_open_dir = os.path.join(hand_dir, 'open')
hand_move_click_dir = os.path.join(hand_dir, 'move_click')
hand_stay_dir = os.path.join(hand_dir, 'stay')

for dir in [hand_dir, hand_open_dir, hand_move_click_dir, hand_stay_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Dictionary to store known faces
known_faces = {}

def save_face(face_img, face_id, photo_count, age, gender):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(face_dir, f'face_{face_id}_photo_{photo_count}_{timestamp}.jpg')
    cv2.imwrite(filename, face_img)
    
    # Save metadata
    with open(os.path.join(face_dir, f'face_{face_id}_metadata.txt'), 'a') as f:
        f.write(f"{timestamp}: Photo {photo_count}, Age: {age}, Gender: {gender}\n")

def estimate_age_gender(face_landmarks):
    # This is a very simple estimation based on face proportions
    face_height = face_landmarks.landmark[152].y - face_landmarks.landmark[10].y
    face_width = face_landmarks.landmark[454].x - face_landmarks.landmark[234].x
    
    ratio = face_width / face_height
    
    if ratio > 0.75:
        gender = "Male"
    else:
        gender = "Female"
    
    # Estimate age based on face proportions (this is very approximate)
    if face_height < 0.2:
        age = "Child (0-12)"
    elif face_height < 0.22:
        age = "Teenager (13-19)"
    elif face_height < 0.24:
        age = "Young Adult (20-35)"
    elif face_height < 0.26:
        age = "Adult (36-60)"
    else:
        age = "Senior (60+)"
    
    return age, gender

def compare_faces(face1, face2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    
    # Resize images to the same size
    gray1 = cv2.resize(gray1, (100, 100))
    gray2 = cv2.resize(gray2, (100, 100))
    
    # Compare histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > 0.8  # Adjust this threshold as needed

def draw_neon_text(img, text, position, font_scale, thickness, text_color, outline_color):
    x, y = position
    # Dibuja el contorno
    cv2.putText(img, text, (x-1, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness*3)
    cv2.putText(img, text, (x+1, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness*3)
    cv2.putText(img, text, (x, y-1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness*3)
    cv2.putText(img, text, (x, y+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness*3)
    # Dibuja el texto principal
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
def draw_face(img, face_landmarks):
    # Dibuja la cara utilizando los puntos de referencia
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * img.shape[1])
        y = int(landmark.y * img.shape[0])
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Dibuja un círculo en cada punto de referencia

def get_rainbow_color(frame_count):
    hue = int(frame_count % 180)
    hsv_color = np.uint8([[[hue, 255, 255]]])
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, rgb_color))

# Añade esta función para detectar el dedo índice levantado
def is_index_finger_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

def has_full_face(face_landmarks):
    # Definir puntos clave para ojos, nariz y boca
    left_eye = face_landmarks.landmark[33]  # Punto del ojo izquierdo
    right_eye = face_landmarks.landmark[263]  # Punto del ojo derecho
    nose_tip = face_landmarks.landmark[4]  # Punta de la nariz
    mouth_left = face_landmarks.landmark[61]  # Esquina izquierda de la boca
    mouth_right = face_landmarks.landmark[291]  # Esquina derecha de la boca

    # Verificar si todos los puntos están dentro del rango visible (0-1)
    points = [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
    return all(0 < point.x < 1 and 0 < point.y < 1 for point in points)

# Añadir variables para la detección de manos
last_hand_positions = []

face_photo_count = {}
next_face_id = 1

hand_photo_count = {}
next_hand_id = 1

frame_count = 0

# Añade estas variables globales
index_finger_up_time = None
last_hand_id = None
last_mouse_position = None

# Mouse control variables
mouse_sensitivity = 1.5
mouse_speed = 1.5
update_rate = 1/60  # 60Hz

prev_mouse_x, prev_mouse_y = pyautogui.position()
last_move_time = time.time()
index_finger_up = False
prev_index_finger_up = False
click_position = None

def is_palm_open(hand_landmarks):
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                   mp_hands.HandLandmark.RING_FINGER_TIP, 
                   mp_hands.HandLandmark.PINKY_TIP]
    finger_pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
                   mp_hands.HandLandmark.RING_FINGER_PIP, 
                   mp_hands.HandLandmark.PINKY_PIP]
    
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(finger_tips, finger_pips))

def is_fist_closed(hand_landmarks):
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                   mp_hands.HandLandmark.RING_FINGER_TIP, 
                   mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    
    return all(hand_landmarks.landmark[tip].y > wrist for tip in finger_tips)

# Añade estas funciones para el control del mouse y la detección de formas de mano
def move_mouse(hand_landmarks):
    global prev_mouse_x, prev_mouse_y, velocity_x, velocity_y
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Mapea las coordenadas de la mano al escritorio
    target_x = np.interp(index_tip.x, [0, 1], [SCREEN_MARGIN, screen_width - SCREEN_MARGIN])
    target_y = np.interp(index_tip.y, [0, 1], [SCREEN_MARGIN, screen_height - SCREEN_MARGIN])
    
    # Calcula la diferencia entre la posición actual y la objetivo
    dx = target_x - prev_mouse_x
    dy = target_y - prev_mouse_y
    
    # Aplica aceleración
    velocity_x = velocity_x * SMOOTHING_FACTOR + dx * (1 - SMOOTHING_FACTOR) * ACCELERATION_FACTOR
    velocity_y = velocity_y * SMOOTHING_FACTOR + dy * (1 - SMOOTHING_FACTOR) * ACCELERATION_FACTOR
    
    # Aplica desaceleración (fricción)
    velocity_x *= DECELERATION_FACTOR
    velocity_y *= DECELERATION_FACTOR
    
    # Aplica el movimiento solo si supera el umbral
    if abs(velocity_x) > MOVEMENT_THRESHOLD or abs(velocity_y) > MOVEMENT_THRESHOLD:
        new_x = max(SCREEN_MARGIN, min(screen_width - SCREEN_MARGIN, prev_mouse_x + velocity_x))
        new_y = max(SCREEN_MARGIN, min(screen_height - SCREEN_MARGIN, prev_mouse_y + velocity_y))
        pyautogui.moveTo(new_x, new_y)
        prev_mouse_x, prev_mouse_y = new_x, new_y

def detect_hand_shape(hand_landmarks):
    if is_palm_open(hand_landmarks):
        return 'open'
    elif is_index_finger_up(hand_landmarks):
        return 'stay'
    elif is_fist_closed(hand_landmarks):
        return 'closed'
    else:
        return 'move_click'

# Modifica la función learn_from_hand_images para contar las imágenes
def learn_from_hand_images():
    hand_images = {'open': 0, 'move_click': 0, 'stay': 0}
    for state in hand_images.keys():
        state_dir = os.path.join(hand_dir, state)
        hand_images[state] = len([f for f in os.listdir(state_dir) if f.endswith('.jpg')])
    
    total_images = sum(hand_images.values())
    print(f"Aprendiendo de {total_images} imágenes de manos:")
    for state, count in hand_images.items():
        print(f"  - {state}: {count} imágenes")
    
    return hand_images

# Llama a esta función al inicio
hand_image_data = learn_from_hand_images()

# Desactiva el failsafe de pyautogui (usa con precaución)
pyautogui.FAILSAFE = False

# Ajusta estas variables según sea necesario
mouse_sensitivity = 1.5
mouse_speed = 1.5

prev_mouse_x, prev_mouse_y = pyautogui.position()
last_hand_shape = None
click_ready = False
stay_position = None
just_clicked = False  # Nueva variable para rastrear si acabamos de hacer clic

# Ajusta estas variables para el control del mouse
SMOOTHING_FACTOR = 0.2  # Ajusta entre 0 y 1 para cambiar la suavidad (más bajo = más suave)
ACCELERATION_FACTOR = 1.5  # Ajusta para cambiar la sensibilidad del movimiento
DECELERATION_FACTOR = 0.95  # Factor de desaceleración (ajusta para cambiar la "fricción")
MOVEMENT_THRESHOLD = 2  # Píxeles mínimos para considerar un movimiento intencional
SCREEN_MARGIN = 10  # Margen en píxeles para los bordes de la pantalla

# Variables para el control del mouse
screen_width, screen_height = pyautogui.size()
prev_mouse_x, prev_mouse_y = pyautogui.position()
velocity_x, velocity_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_shape = detect_hand_shape(hand_landmarks)
            
            if hand_shape == 'open':
                move_mouse(hand_landmarks)
                click_ready = False
                stay_position = None
                just_clicked = False
            elif hand_shape == 'stay':
                if stay_position is None:
                    stay_position = pyautogui.position()
                click_ready = True
            elif hand_shape == 'move_click':
                if last_hand_shape == 'stay' and not just_clicked:
                    if stay_position:
                        pyautogui.moveTo(stay_position[0], stay_position[1])
                    pyautogui.click()
                    just_clicked = True
                elif not just_clicked:
                    move_mouse(hand_landmarks)
                click_ready = False
            elif hand_shape == 'closed' and click_ready and not just_clicked:
                pyautogui.click()
                just_clicked = True
                click_ready = False
            
            # Guarda la imagen de la mano y actualiza el conteo
            h, w, _ = frame.shape
            x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w))
            x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w))
            y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h))
            y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h))
            
            if x_min < x_max and y_min < y_max:
                hand_img = frame[y_min:y_max, x_min:x_max]
                # save_hand(hand_img, hand_shape)
                if hand_shape in hand_image_data:
                    hand_image_data[hand_shape] += 1

            # Muestra el estado de la mano y el conteo de imágenes
            cv2.putText(frame, f"Hand shape: {hand_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Images: Open {hand_image_data['open']}, Stay {hand_image_data['stay']}, Move {hand_image_data['move_click']}, Closed {hand_image_data.get('closed', 0)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            last_hand_shape = hand_shape

            # Obtener la posición de la mano
            hand_position = [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y]
            last_hand_positions.append(hand_position)

            # Comprobar si hay dos manos en la pantalla
            if len(last_hand_positions) == 2:
                distance = np.linalg.norm(np.array(last_hand_positions[0]) - np.array(last_hand_positions[1]))
                if distance < 0.1:  # Ajusta este umbral según sea necesario
                    cv2.putText(frame, "¡Alerta! Manos juntas", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Limitar el número de posiciones almacenadas
            if len(last_hand_positions) > 2:
                last_hand_positions.pop(0)

    cv2.imshow('Hand Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()
