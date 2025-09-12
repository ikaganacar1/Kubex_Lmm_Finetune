import os
import json
import time
import google.genai as genai


api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = "AIzaSyDq-1M6b58w3mg0Yj20UKqhwtstXvkVSS8" 

output_file = 'SimpleSingleStepOperations.jsonl'
num_examples_to_generate = 1000 # Set the total number of examples you want

model_name = "gemini-2.5-flash"

prompt = """
Generate a complete, multi-turn training example in the specified JSON format for a simple, single-step Kubernetes operation.

The final output MUST be a single raw JSON object, starting with `{` and ending with `}`. Do not include markdown formatting like ```json.

### JSON Structure Requirements:
1.  The root is a JSON object with a single key: `"messages"`.
2.  The `"messages"` array must contain this exact sequence of 5 roles:
    - `system`: A static introductory message defining the assistant's role.
    - `user`: Asks for a simple, single-step Kubernetes operation.
    - `assistant`: Gives a brief acknowledgment and makes a SINGLE tool call.
    - `tool`: Provides a realistic JSON response from the tool.
    - `assistant`: Provides the final, user-friendly formatted answer based on the tool's response.

### Content Requirements:
-   **system message**: Use the exact content provided in the example below.
-   **user message**: The request should be varied. Use patterns like "List all [resource]", "Get status for [deployment-name]", "Show me pods in [namespace]", "Check the health of [service]".
-   **tool_calls**: The function name should be specific to the action (e.g., `cluster_index`, `list_pods`, `get_deployment_status`).
-   **tool content**: The content must be a JSON string representing a realistic API response for the tool call.
-   **final assistant message**: This should be a nicely formatted, human-readable summary of the information from the tool content, often using markdown lists or tables.

### Perfect Example Format (Adhere to this structure strictly):
```json
{"messages": [{"role": "system", "content": "You are a Kubernetes management assistant. You have access to the KubeX API to manage clusters, deployments, pods, and other Kubernetes resources. Only use the provided API tools. If asked about topics outside Kubernetes management, politely decline."}, {"role": "user", "content": "List all clusters"}, {"role": "assistant", "content": "I'll retrieve the list of all Kubernetes clusters for you.", "tool_calls": [{"type": "function", "function": {"name": "cluster_index", "arguments": "{}"}}]}, {"role": "tool", "content": "{\"clusters\": [{\"id\": \"prod-k8s-01\", \"name\": \"Production Primary\", \"favorite\": true, \"status\": \"healthy\"}, {\"id\": \"staging-k8s\", \"name\": \"Staging\", \"favorite\": false, \"status\": \"healthy\"}]}"}, {"role": "assistant", "content": "I found 2 Kubernetes clusters:\n\n1. **Production Primary** (prod-k8s-01)\n   - Status: ✅ Healthy\n   - Favorite: ⭐ Yes\n\n2. **Staging** (staging-k8s)\n   - Status: ✅ Healthy\n   - Favorite: No"}]}
```

Now, generate a new, unique example following all the rules above.
"""

def generate_data():
    """
    Main function to generate synthetic data and write it to a file.
    """
    if api_key == "YOUR_API_KEY":
        print("🛑 ERROR: Please replace 'YOUR_API_KEY' with your actual Google AI API key.")
        return

    print("Configuring generative AI client...")
    client = genai.Client(api_key=api_key)
    

    print(f"Starting generation of {num_examples_to_generate} examples into '{output_file}'...")

    # Open the file in append mode ('a') to add new lines without deleting existing ones
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in range(num_examples_to_generate):
            try:
                print(f"Generating example {i + 1}/{num_examples_to_generate}...")
                
                # Make the API call with retry logic for rate limiting
                response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                ) 
                
                # Clean up the response to ensure it's valid JSON
                clean_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()

                # Validate that the response is a single, complete JSON object
                if not (clean_text.startswith('{') and clean_text.endswith('}')):
                    raise json.JSONDecodeError("Response is not a complete JSON object.", clean_text, 0)

                # Parse the JSON to ensure it's valid before writing
                data = json.loads(clean_text)
                
                # Convert the Python dictionary back to a compact JSON string (one line)
                json_line = json.dumps(data, ensure_ascii=False)
                
                # Write the single-line JSON to the file, followed by a newline character
                f.write(json_line + '\n')
                f.flush() # Ensure data is written to disk immediately

            except json.JSONDecodeError as e:
                print(f"  -> ⚠️ WARNING: Failed to decode JSON for example {i + 1}. Skipping. Error: {e}")
                # print(f"     Received text: {clean_text}") # Uncomment for debugging
            
                
            except Exception as e:
                print(f"  -> 🛑 An unexpected error occurred on example {i + 1}: {e}. Skipping.")

    print(f"\n✅ Generation complete. Data saved to '{output_file}'.")

if __name__ == "__main__":
    generate_data()
