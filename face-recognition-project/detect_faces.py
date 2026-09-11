import cv2
from insightface.app import FaceAnalysis
def main():
    app = FaceAnalysis(
        name = "buffalo_l",
        providers= ["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id = 0,
        det_size=(640,640)
    )

    camera =cv2.VideoCapture(0)
    if not camera.isOpened():
        print("error: could not open webcam.")
        return
    
    print("camera started.")
    print("press q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("error: could not read frame.")
            break
        faces = app.get(frame)

        for face in faces:
            x1,y1,x2,y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            confidence = face.det_score

            text = f"Face: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1- 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Faces detected: {len(faces)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
            cv2.imshow("Face detection", frame)

            if cv2.waitKey(1) & 0xFF ==  ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()