import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis


def normalize(embedding):
    return embedding / np.linalg.norm(embedding)


def build_centroids(known_faces):
    """
    Group embeddings by person and create one centroid
    for each person.
    """

    person_embeddings = {}

    for face in known_faces:

        name = face["name"]
        embedding = face["embedding"]

        if name not in person_embeddings:
            person_embeddings[name] = []

        person_embeddings[name].append(embedding)

    centroids = {}

    for name, embeddings in person_embeddings.items():

        embeddings = np.array(embeddings)

        # Average all embeddings for this person
        centroid = np.mean(embeddings, axis=0)

        # Normalize the centroid
        centroid = normalize(centroid)

        centroids[name] = centroid

        print(
            f"Created centroid for {name} "
            f"using {len(embeddings)} images."
        )

    return centroids


def cosine_similarity(a, b):

    a = normalize(a)
    b = normalize(b)

    return np.dot(a, b)


def recognize_face(embedding, centroids):

    scores = []

    for name, centroid in centroids.items():

        similarity = cosine_similarity(
            embedding,
            centroid
        )

        scores.append(
            (name, similarity)
        )

    # Sort from highest similarity to lowest
    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_name, best_similarity = scores[0]

    # If there is another person, calculate margin
    if len(scores) > 1:

        second_similarity = scores[1][1]

        margin = (
            best_similarity -
            second_similarity
        )

    else:

        second_similarity = -1.0
        margin = 1.0

    # Recognition thresholds
    similarity_threshold = 0.45
    margin_threshold = 0.10

    if (
        best_similarity >= similarity_threshold
        and margin >= margin_threshold
    ):
        recognized_name = best_name

    else:
        recognized_name = "Unknown"

    return (
        recognized_name,
        best_similarity,
        second_similarity,
        margin
    )


def main():

    print("Loading face recognition model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    print("Loading saved embeddings...")

    with open(
        "models/face_embeddings.pkl",
        "rb"
    ) as file:

        known_faces = pickle.load(file)

    print(
        f"Loaded {len(known_faces)} face embeddings."
    )

    # Build one centroid per person
    centroids = build_centroids(
        known_faces
    )

    print(
        f"Total people in database: "
        f"{len(centroids)}"
    )

    print("Opening webcam...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    print("Camera started.")
    print("Press 'q' to quit.")

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break

        faces = app.get(frame)

        for face in faces:

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            (
                name,
                similarity,
                second_similarity,
                margin
            ) = recognize_face(
                face.embedding,
                centroids
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name} ({similarity:.2f})",
                (x1, y1 - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Second: {second_similarity:.2f}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Margin: {margin:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Margin: {margin:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Face Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()