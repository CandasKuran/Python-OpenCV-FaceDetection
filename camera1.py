import cv2
import numpy as np

# pour importer le cascade
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

# les photos et les noms des personnes à reconnaître
img1 = cv2.imread('./images/photo1.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/photo2.png',cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('./images/photo3.png',cv2.IMREAD_GRAYSCALE)


known_faces = [img1, img2, img3]
known_names = ['candas', 'fateme', 'flavio']

# parametre de camera
camera_device = 0
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

# traitement des images et reconnaissance des visages
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    # augmenter le contraste par l'égalisation des niveaux de gris et de l'histogramme de l'image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # yüz tespiti
    faces = face_cascade.detectMultiScale(gray)

    # processus pour chaque face
    for (x, y, w, h) in faces:
        
        # prendre la zone du visage
        face_roi = gray[y:y+h, x:x+w]

        # connaisance de face
        match_result = None
        for i, known_face in enumerate(known_faces):
            try:
            # comparer les visages
                match = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
            
            except Exception as e:
                print(f"il y a unnnnnnn problem {e}")
                continue

            # Comme dans le message d'erreur, "The truth value of an array with more than one element is ambiguous" 
            # est dû au fait que l'un des deux tableaux numpy que vous comparez comporte plus d'un élément. 
            # Par conséquent, avant d'effectuer la comparaison, vous devez convertir les tableaux numpy 
            # en une seule valeur numérique. Pour ce faire, vous devrez peut-être convertir les variables match_result
            # et match à la valeur maximale à l'aide de la fonction np.max() :
            match = np.max(match)
            if match_result is None or match > match_result:
                match_result = match
                best_match_index = i

        # si nous avons trouvé la meilleure correspondance, 
        # nous le montrons en traçant un rectangle autour du visage et en écrivant le nom en dessous.

            try:
        
                if match_result > 0.5:
                    name = known_names[best_match_index]
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    frame = cv2.putText(frame, name, (x, y+h+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

                # si nous n'avons pas trouvé de correspondance, 
                # nous le montrons en traçant un rectangle rouge autour du visage et en écrivant 'Inconnu' en dessous.
                if match_result is None or match_result < 0.5:
                    name = 'Inconnu'
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    frame = cv2.putText(frame, name, (x, y+h+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
                

            except Exception as e:
                print(f"error {e}")
    # affichage image
    cv2.imshow('Face Recognition', frame)

    # press 'q' pour quitter le program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break