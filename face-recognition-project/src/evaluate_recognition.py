import cv2
import pickle
import numpy as np
import csv
import os
from insightface.app import FaceAnalysis


def normalize(embedding):
    return embedding / np.linalg.norm(embedding)


def build_centroids(known_faces):
    person_embeddings = {}

    for face in known_faces:
        name = face["name"]

        if name not in person_embeddings:
            person_embeddings[name] = []

        person_embeddings[name].append(face["embedding"])

    centroids = {}

    for name, embeddings in person_embeddings.items():
        centroid = np.mean(
            np.array(embeddings),
            axis=0
        )

        centroids[name] = normalize(centroid)

    return centroids


def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)

    return float(np.dot(a, b))


def main():

    print("===================================")
    print(" Face Recognition Evaluation")
    print("===================================")
    print()

    label = input(
        "Who is in front of the camera? "
        "(sumeet/unknown): "
    ).strip().lower()

    if label not in ["sumeet", "unknown"]:
        print("Please enter 'sumeet' or 'unknown'.")
        return

    try:
        num_samples = int(
            input("Number of samples (default 20): ") or "20"
        )
    except ValueError:
        print("Invalid number.")
        return

    print()
    print("Loading face recognition model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    print("Loading embeddings...")

    with open(
        "models/face_embeddings.pkl",
        "rb"
    ) as file:
        known_faces = pickle.load(file)

    centroids = build_centroids(known_faces)

    print(
        f"Loaded {len(known_faces)} embeddings."
    )

    print(
        f"Known people: {list(centroids.keys())}"
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print()
    print("Camera started.")
    print()
    print("Look at the camera.")
    print("Move your face naturally.")
    print("Samples will be collected automatically.")
    print()

    similarities = []

    while len(similarities) < num_samples:

        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        faces = app.get(frame)

        if len(faces) > 0:

            # Select largest face
            face = max(
                faces,
                key=lambda x:
                (x.bbox[2] - x.bbox[0]) *
                (x.bbox[3] - x.bbox[1])
            )

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            similarity = cosine_similarity(
                face.embedding,
                centroids["sumeet"]
            )

            similarities.append(similarity)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Sample: {len(similarities)}/{num_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Similarity: {similarity:.3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Recognition Evaluation",
            frame
        )

        # Small delay between samples
        key = cv2.waitKey(150) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    if len(similarities) == 0:
        print("No face samples collected.")
        return

    similarities = np.array(similarities)

    print()
    print("===================================")
    print(" Results")
    print("===================================")

    print(f"Label:        {label}")
    print(f"Samples:      {len(similarities)}")
    print(f"Minimum:      {similarities.min():.3f}")
    print(f"Maximum:      {similarities.max():.3f}")
    print(f"Average:      {similarities.mean():.3f}")
    print(f"Std deviation:{similarities.std():.3f}")

    # Save results
    os.makedirs("models", exist_ok=True)

    output_file = "models/evaluation_results.csv"

    file_exists = os.path.exists(output_file)

    with open(
        output_file,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "label",
                "sample",
                "similarity"
            ])

        for i, similarity in enumerate(
            similarities,
            start=1
        ):

            writer.writerow([
                label,
                i,
                f"{similarity:.6f}"
            ])

    print()
    print(
        f"Results saved to: {output_file}"
    )


if __name__ == "__main__":
    main()