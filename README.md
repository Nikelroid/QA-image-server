# Vision QA Server

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108%2B-009688?logo=fastapi&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Vision_API-4285F4?logo=google-cloud&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)

## Description

**Vision QA Server** is a robust backend application designed to "see" and understand images to answer natural language questions. Built with **FastAPI**, it leverages the **Google Cloud Vision API** to extract comprehensive metadata (objects, text, faces, colors, landmarks) and uses a custom internal heuristic engine to interpret user questions and provide context-aware answers.

Unlike simple API wrappers, this project implements a `QuestionAnalyzer` that categorizes intent (e.g., counting objects, identifying colors, reading text) to synthesize human-like responses. It supports both stateless REST calls and stateful WebSocket connections for real-time applications.

## Features

  * **Smart Question Analysis:** Automatically categorizes questions into types such as `count`, `identify`, `read_text`, `color`, `location`, and `yes_no` to generate relevant answers.
  * **Comprehensive Image Analysis:** Detects objects, labels, text (OCR), faces (with emotion), landmarks, logos, and dominant colors.
  * **Real-Time WebSockets:** Full WebSocket support for continuous interaction, including connection management and live status updates.
  * **Cloud Native:** deeply integrated with Google Cloud Platform:
      * **Vision API:** For core image intelligence.
      * **Cloud Storage:** Automatically uploads and hosts analyzed images.
      * **Secret Manager:** Securely manages credentials for production deployments.
  * **Production Ready:** Includes a Dockerfile for containerization and shell scripts for streamlined deployment to Google Cloud Run.

## Installation

### Prerequisites

  * Python 3.11+
  * Google Cloud Platform Account with Vision API enabled.
  * Google Cloud SDK (`gcloud`) installed (for deployment).

### Local Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/nikelroid/qa-image-server.git
    cd qa-image-server
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Dependencies include `fastapi`, `uvicorn`, `google-cloud-vision`, and `pydantic`)*

4.  **Configure Credentials:**
    You must set the `GOOGLE_CREDENTIALS_JSON` environment variable or place your Service Account JSON in the root directory and reference it.

    ```bash
    export GOOGLE_CREDENTIALS_JSON='{...your_service_account_json...}'
    # OR
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-key.json"
    ```

## Usage

### Running Locally

Start the server using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080`.

### API Endpoints

  * **POST** `/analyze-image`: Send a base64 encoded image and a question.
    ```json
    {
      "image": "base64_encoded_string_here...",
      "question": "How many cars are in this image?"
    }
    ```
  * **GET** `/health`: Check service status and Cloud connections.
  * **WS** `/ws/{client_id}`: Connect via WebSocket for streamed analysis.

### Deployment

The project includes a streamlined deployment script for **Google Cloud Run**.

1.  Edit `deploy.sh` to set your `PROJECT_ID`, `REGION`, and `BUCKET_NAME`.
2.  Run the script:
    ```bash
    chmod +x deploy.sh
    ./deploy.sh
    ```
    This script handles project configuration, Secret Manager permissions, Docker builds, and Cloud Run deployment automatically.

## Contributing

Contributions are welcome\!

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewFeature`).
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.

## License

Distributed under the Apache License, Version 2.0. See `LICENSE` for more information.

## Contact

For support or inquiries, please open an issue in the GitHub repository.
