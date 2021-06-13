import cv2
import sys

# Pasa los nombres de la imagen y la cascada como argumentos de línea de comandos
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml" #sys.argv[2]

# Creamos la cascada y la inicializamos con nuestra cascada de caras
faceCascade = cv2.CascadeClassifier(cascPath)

# Leemos la imagen y la convertimos a escala de grises.
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecta la cara real en la imagen
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(25,25),
    maxSize=(200,200)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print ("Found {0} faces!".format(len(faces)))

# Dibuja un  rectángulo alrededor de los rostros
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
cv2.imshow("Faces found", image)
cv2.waitKey(0) 