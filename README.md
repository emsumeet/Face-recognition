# Face Recognition System using InsightFace and OpenCV

A real-time face recognition system built in Python using **InsightFace**, **OpenCV**, and **cosine similarity**.

The project detects faces from a webcam or image, generates facial embeddings using a pretrained deep-learning model, stores embeddings for known individuals, and performs real-time identity recognition.

The project is being developed incrementally with a focus on understanding the complete face-recognition pipeline rather than simply using a prebuilt recognition API.

---

## Features

- Real-time face detection using a webcam
- Face embedding generation using InsightFace
- Multiple reference images per person
- L2-normalized facial embeddings
- Centroid-based identity representation
- Cosine-similarity matching
- Unknown-person rejection
- Best-match and second-best-match comparison
- Recognition margin calculation
- Automated recognition evaluation
- Multi-person face detection
- Evaluation results saved to CSV

---

## How It Works

The recognition pipeline is:

```text
Image / Webcam
       ↓
Face Detection
       ↓
Face Alignment
       ↓
InsightFace Embedding
       ↓
L2 Normalization
       ↓
Identity Centroids
       ↓
Cosine Similarity
       ↓
Best Match + Second Best Match
       ↓
Similarity & Margin Thresholds
       ↓
Known Person / Unknown
```

InsightFace converts each detected face into a numerical embedding representing facial features.

Multiple embeddings belonging to the same person are averaged to create an identity **centroid**.

During recognition, the embedding of the detected face is compared with the stored centroids using cosine similarity.

---

## Recognition Logic

For every detected face, the system calculates:

```text
Best similarity
Second-best similarity
Margin = Best similarity - Second-best similarity
```

A person is recognized only when the similarity and margin satisfy the configured thresholds.

This helps reduce false matches compared with simply selecting the identity with the highest similarity score.

---

## Project Structure

```text
face-recognition-project/
│
├── data/
│   └── faces/
│       ├── sumeet/
│       └── person 2/
│
├── models/
│   ├── face_embeddings.pkl
│   └── evaluation_results.csv
│
├── src/
│   ├── detect_faces.py
│   ├── create_embeddings.py
│   ├── recognize_face.py
│   └── evaluate_recognition.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

Generated embeddings and evaluation files can be excluded from Git using `.gitignore`.

---

## Technologies

- Python 3
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy
- Pickle
- CSV

### Face Model

The project currently uses the InsightFace:

```text
buffalo_l
```

model package.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/emsumeet/Face-recognition.git
cd face-recognition-project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Adding a Person

> **Note:** Sample face images are not included in this repository for privacy reasons.  
> Add your own sample photos inside the `data/faces/` directory before generating embeddings.

Create a directory inside:

```text
data/faces/
```

For example:

```text
data/faces/sumeet/
```

Add several clear images of the person's face:

```text
data/faces/sumeet/
├── image1.jpeg
├── image2.jpeg
├── image3.jpeg
└── ...
```

Using images with different angles, expressions, lighting conditions, and distances can help produce a more representative identity embedding.

---

## Generate Face Embeddings

Run:

```bash
python src/create_embeddings.py
```

The program:

1. Scans the face dataset
2. Detects faces in each image
3. Selects the primary face
4. Extracts a facial embedding
5. L2-normalizes the embedding
6. Saves the embeddings

The generated database is stored at:

```text
models/face_embeddings.pkl
```

---

## Real-Time Face Recognition

Run:

```bash
python src/recognize_face.py
```

The webcam will open and detected faces will be compared against the enrolled identities.

The interface displays information such as:

```text
sumeet (0.68)
Second: 0.24
Margin: 0.44
```

Press:

```text
q
```

to quit.

---

## Evaluation

The project includes an evaluation script for collecting recognition measurements from webcam samples.

Run:

```bash
python src/evaluate_recognition.py
```

The evaluator records metrics including:

```text
Actual identity
Predicted identity
Best similarity
Second-best similarity
Recognition margin
Accuracy
```

Evaluation results are stored in:

```text
models/evaluation_results.csv
```

These measurements can be used to determine better similarity and margin thresholds instead of selecting thresholds arbitrarily.

---

## Current Progress

### Stage 1 — Face Detection

Completed.

Real-time webcam face detection using InsightFace and OpenCV.

### Stage 2 — Face Embeddings

Completed.

Multiple reference images are converted into normalized facial embeddings.

### Stage 3 — Face Recognition

In progress.

Implemented:

- Cosine similarity
- L2-normalized embeddings
- Identity centroids
- Multiple enrolled identities
- Unknown-person rejection
- Best/second-best comparison
- Recognition margin
- Automated evaluation

Current work focuses on **threshold calibration and reducing false positives**.

---

## Preliminary Evaluation

Initial testing demonstrated a clear difference between genuine and non-matching face comparisons, while also revealing that individual frames can occasionally produce high similarity scores for the wrong identity.

This motivates using both:

```text
Similarity threshold
+
Recognition margin
```

rather than relying only on the highest cosine-similarity score.

More extensive evaluation is planned before reporting final performance metrics.

---

## Planned Improvements

Future development will include:

- Larger evaluation dataset
- Similarity-threshold calibration
- Margin-threshold calibration
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- ROC analysis
- Additional enrolled identities
- Recognition stability across consecutive video frames
- Improved handling of lighting and pose variation
- Better dataset collection
- Visualization of evaluation results
- Performance benchmarking

---

## What I Learned

This project explores several practical machine-learning and computer-vision concepts:

- Face detection
- Deep-learning embeddings
- Feature vectors
- Vector normalization
- Cosine similarity
- Centroid representations
- Classification thresholds
- Open-set recognition
- False positives and false negatives
- Model evaluation

A major goal of the project is understanding how pretrained neural-network representations can be incorporated into a complete recognition and evaluation pipeline.

---

## Privacy

Facial embeddings and personal training images should be handled carefully.

The repository should avoid publishing private face datasets or generated embedding files unless the individuals involved have explicitly agreed to their publication.

---

## Author

**Sumeet Varghade**

Computer Science graduate interested in Artificial Intelligence, Machine Learning, Computer Vision, Retrieval-Augmented Generation (RAG), and applied AI research.

GitHub: **emsumeet**

---

## Status

**Active Development**

The basic face-recognition pipeline is operational. Current development focuses on systematic evaluation, threshold calibration, and improving recognition reliability.
