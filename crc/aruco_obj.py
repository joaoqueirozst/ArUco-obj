import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import imutils
import time

# Carrega o dicionario que foi usado para gerar os ArUcos e
# inicializa o detector usando valores padroes para os parametros
parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)

vs = VideoStream(src=0).start()
time.sleep(2.0)

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename) as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith('f'):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)
    return np.array(vertices), faces

def rotate_obj(vertices, angle_degrees): # mtx rotação obj
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Matriz de rotação em torno do eixo Y
    rotation_mtx = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    # Aplica a rotação a todos os vértices
    rotated = vertices @ rotation_mtx.T
    return rotated

vertices, faces = load_obj("Penguin.obj")
vertices = vertices * 0.001
marker_length = 0.25

dz = 0
ids = [17, 35, 49, 62, 68, 88, 94, 106, 110, 118, 973] # 17, 35, 49, 62, 68, 88, 94, 106, 110, 118, 973
estado = 0
sequencia = False
fimzao = False
while(True):
    frame = vs.read()
    frame = imutils.resize(frame, width=1024)
    height, width = frame.shape[:2]

    # Matriz de câmera simulada
    focal_length = width
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1))

    # Detecta os marcadores na imagem
    markerCorners, markerIds, _ = arucoDetector.detectMarkers(frame)
    print(markerCorners)

    if markerIds is not None:
        markerIds_flat = markerIds.flatten().tolist()

        # Ativa a sequência apenas quando o primeiro marcador da lista for detectado
        if not sequencia:
            if ids[0] in markerIds_flat:
                sequencia = True
                estado = 0
                dz = 0

        if sequencia:
            id_esperado = ids[estado]               

            if id_esperado in markerIds_flat:
                i = list(markerIds_flat).index(id_esperado)
                corners = [markerCorners[i]]
                ids_m = np.array([[id_esperado]])

                cv2.aruco.drawDetectedMarkers(frame, corners, ids_m)

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

                for rvec, tvec in zip(rvecs, tvecs):
                    # Desenhar eixo de referência no ArUco
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                    r_mtx, _ = cv2.Rodrigues(rvec) # matriz de rotação -> girar vetores do referencial local do marcador para ref. global da cam
                    
                    if estado in [0, 1, 2, 3, 8, 9]:
                        transfer = np.array([[0], [-dz], [0]])
                        if dz < 0.1:
                            dz += 0.002
                        angle = 0
                    elif estado in [10]:
                        transfer = np.array([[0], [-dz], [0]])
                        angle = 0
                        if dz < 0:
                            dz += 0.002
                        else:
                            fimzao = True

                    elif estado in [4]:  
                        transfer = np.array([[dx], [-dz], [0]])
                        if dz < 0:
                            dz += 0.002
                            angle = 0 
                        elif dx < 0.1:
                            dx += 0.002
                            angle = 90 
                    elif estado in [5, 6]:  
                        transfer = np.array([[dx], [0], [0]])
                        if dx < 0.1:
                            dx += 0.002
                        angle = 90 
                    elif estado in [7]: 
                        transfer = np.array([[dx], [dz], [0]])
                        if dx < 0:
                            dx += 0.002
                            angle = 90 
                        elif dz < 0.1:
                            dz += 0.002
                            angle = 180  
                    else:
                        transfer = np.array([[0], [-dz], [0]])
                        angle = 0

                    # Rotaciona os vértices do modelo para o sentido atual
                    rotated_vertices = rotate_obj(vertices, angle)                                                               

                    rotation = r_mtx@transfer
                    t = tvec.T + rotation

                    imgpts, _ = cv2.projectPoints(rotated_vertices, rvec, t, camera_matrix, dist_coeffs)
                    imgpts = imgpts.reshape(-1, 2).astype(int)

                    # Desenha as faces do modelo
                    limiar = 0.001
                    for face in faces:
                        pts = imgpts[face]
                        values = [rotated_vertices[i][2] for i in face] #
                        media = np.mean(values) #
                        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0), thickness=1)
                        # if media < limiar: #
                        #     cv2.fillPoly(frame, [pts], color=(0, 255, 255)) #
                        # else:    
                        #     cv2.fillPoly(frame, [pts], color=(0, 0, 0))

                if estado == 0:
                    if dz >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 1
                        dz = -0.1
                elif estado == 1:
                    if dz >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 2
                        dz = -0.1
                elif estado == 2:
                    if dz >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 3
                        dz = -0.1
                elif estado == 3:
                    if dz >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 4
                        dz = -0.1
                        dx = 0
                elif estado == 4:
                    if dx >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 5
                        dx = -0.1
                elif estado == 5:
                    if dx >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 6        
                        dx = -0.1
                elif estado == 6:
                    if dx >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 7
                        dz = 0
                        dx = -0.1
                elif estado == 7:
                    if dz >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 8
                        dz = -0.1
                elif estado == 8:
                    if dz >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 9
                        dz = -0.1
                elif estado == 9:
                    if dz >= 0.1 and (ids[estado+1] in markerIds_flat):
                        estado = 10
                        dz = -0.1                                                                              
                # if estado + 1 < len(ids) and ids[estado + 1] in markerIds_flat:
                #     estado += 1

            if not any(id_valid in markerIds_flat for id_valid in ids):
                sequencia = False
                estado = 0
                dz = 0

    else:
        sequencia = False
        estado = 0
        dz = 0           

    if fimzao == False:
        cv2.imshow('Aruco_obj',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # mostrar = cv2.imread("image_penguin.jpeg")  
        # cv2.imshow('Aruco_obj',mostrar)
        # cv2.waitKey(0)
        break

cv2.destroyAllWindows()
vs.stop()
