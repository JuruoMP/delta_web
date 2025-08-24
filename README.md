# Delta Web

This is a web application that provides audio transcription and speaker diarization services.

## Environment Setup

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**

    Before installing the Python packages, you need to install `ffmpeg`.

    *   **On macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```

    *   **On Debian/Ubuntu:**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```

    Then, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables:**

    Create a `.env` file in the root directory of the project and add the following environment variables:

    ```
    HF_TOKEN=your_hugging_face_token
    ```

    You can obtain a Hugging Face token from [here](https://huggingface.co/settings/tokens).

## How to Run

1.  **Start the application:**

    ```bash
    ./run.sh
    ```

    This will start the Flask application using Gunicorn.

2.  **Access the application:**

    Open your web browser and go to `http://127.0.0.1:8000`.

## How to Run Tests

To run the test script for the Whisper ASR service, use the following command:

```bash
python test_whisper_service.py
```