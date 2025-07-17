# Case 3 Bithealth AI Pre Test : Hospital Triage System

A FastAPI service to recommend medical departments based on patient symptoms using Google’s Gemini LLM via LangChain.

## Prerequisites
- Python 3.9+
- Google AI Studio API key (store in `.env` file)
- Visual Studio Code with the REST Client extension (for testing)

## Obtaining a Google AI Studio API Key
1. Visit [Google AI Studio](https://ai.google.dev/).
2. Sign in with your Google account.
3. Navigate to the API section and create a new API key:
   - Click "Get API key" or similar in the Google AI Studio dashboard.
   - Follow prompts to generate a key for the Gemini API.
4. Copy the API key and store it in a `.env` file in the project root (see Setup Instructions).
5. Ensure the key has access to the Gemini models.

## Setup Instructions
1. Clone or download the project:
   ```bash
   git clone https://github.com/JustAthar19/Case3_BitHealth_AI_PreTest.git
   cd case3_triage_system
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv case3_env
   source case3_env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Google AI Studio API key:
   ```bash
   echo "GOOGLE_API_KEY=your_api_key" > .env
   ```

## Running the Application
1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
2. The server runs at `http://localhost:8000`.

## Testing the Endpoint
### Using Swagger UI
- Open `http://localhost:8000/docs` in a browser.
- Use the interactive UI to send POST requests to `/recommend`.

### Using curl
- Send a POST request:
  ```bash
  curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"gender": "female", "age": 22, "symptoms": ["pusing", "mual", "sulit berjalan"]}'
  ```
- Expected response:
  ```json
  {
      "recommended_department": "Neurology"
  }
  ```

### Using REST Client in VS Code
1. Install the REST Client extension in VS Code:
   - Open the Extensions view (`Ctrl+Shift+X`).
   - Search for "REST Client" (by Huachao Mao) and install.
2. Create a file named `test.http` in the project root.
3. Add the following to `test.http`:
   ```
   POST http://localhost:8000/recommend
   Content-Type: application/json

   {
       "gender": "female",
       "age": 22,
       "symptoms": ["pusing", "mual", "sulit berjalan"]
   }
   ```
4. Open `test.http` in VS Code, click "Send Request" above the request, and view the response.
5. Expected response:
   ```json
   {
       "recommended_department": "Neurology"
   }
   ```

## Example Request
```json
{
    "gender": "female",
    "age": 22,
    "symptoms": ["pusing", "mual", "sulit berjalan"]
}
```

## Notes
- The service uses Google’s Gemini LLM via LangChain. If the LLM fails, it falls back to a rule-based system.
- Ensure the `.env` file is not committed to version control (included in `.gitignore`).
- The `urllib3` dependency is pinned to `<2` to avoid LibreSSL issues on macOS.
