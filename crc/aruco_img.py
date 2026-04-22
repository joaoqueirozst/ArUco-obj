import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import imutils
import time

# source ./env_aruco/bin/activate

# Carrega o dicionario que foi usado para gerar os ArUcos e
# inicializa o detector usando valores padroes para os parametros
parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Detecta os marcadores na imagem
vs = VideoStream(src=0).start()
time.sleep(2.0)

while(True):

    frame = vs.read()
    frame = imutils.resize(frame, width=1024)

    # Detecta os marcadores na imagem
    markerCorners, markerIds, rejectedImgPoints = arucoDetector.detectMarkers(frame)
    # print(markerCorners)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Desenha as quinas detectadas na imagem
    img01_corners = cv2.aruco.drawDetectedMarkers(frame_rgb, markerCorners, markerIds)
    # cv2.imshow('img01_corners',img01_corners)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    img = cv2.imread('') # inserir imagem
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tamanho da imagem que sera inserida no lugar dos ArUcos
    [l,c,ch] = np.shape(img_rgb)
    # Pixels das quinas da imagem que sera inserida com ajuda do warp
    pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])
    if markerIds is not None:
        if len(markerIds) > 0:
            for mark in markerCorners:  # Para cada marcador detectado
                # Anota as quinas do marcador detectado como pontos de destino da homografia
                pts_dst = np.array(mark[0])

                # Calcula a homografia
                H, status = cv2.findHomography(pts_src, pts_dst)

                # Faz o warp na imagem para que ela seja inserida
                warped_image = cv2.warpPerspective(img_rgb, H, (frame_rgb.shape[1],frame_rgb.shape[0]))
                # plt.figure(figsize=[10,10])
                # plt.imshow(warped_image)

                # Prepara a mascara para que apenas a foto contida no warp da imagem substitua pixels da outra imagem
                mask = np.zeros([frame_rgb.shape[0], frame_rgb.shape[1]], dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32([pts_dst]), (1, 1, 1), cv2.LINE_AA)

                # Transforma essa mascara em 3 canais
                mask3 = np.zeros_like(warped_image)
                for i in range(0, 3):
                    mask3[:,:,i] = mask

                print(mark[0])
                rolls = mark[0][:,1]
                cols = mark[0][:,0]

                if (np.all(cols<1024) and np.all(rolls<768)):
                    frame_masked = cv2.multiply(frame_rgb, 1-mask3)
                    frame_rgb = cv2.add(warped_image, frame_masked)
            
    frame_nova = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

    height, width = frame_nova.shape[:2]

    cv2.imshow('Aruco',frame_nova)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
