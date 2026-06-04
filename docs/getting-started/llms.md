# LLM Configuration

The LLM config file allows you to select an LLM model of interest and provide API key and other parameters of interest. We provide an example file `llm_config.example.yml` at the base of MethylCurate's directory.

## Configuration Overview

THe following table provides an overview of the variables that can be configured in the LLM config file.

| Parameter         | Values                                    | Description                                                                                                                                                               |
|-------------------|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| provider          | {openai, azure_openai, anthropic, ollama} | This describes the LLM model provider, must be one of the listed values.                                                                                                  |
| model             | string                                    | This describes the specific LLM model being used.                                                                                                                         |
| api_key           | string                                    | API key for the selected provider.                                                                                                                                        |
| base_url          | URL                                       | Override the base URL for the provider (e.g., proxies, alternative endpoints, or Ollama host). Leave unset to use the default.                                            |
| azure_endpoint    | URL                                       | Your Azure endpoint                                                                                                                                                       |
| azure_deployment  | string                                    | Your Azure deployment                                                                                                                                                     |
| azure_api_version | string                                    | Your Azure API version                                                                                                                                                    |
| temperature       | float                                     | The temperature of the model. Increasing the temperature will make the model answer more creatively.                                                                      |
| top_k             | integer                                   | Limits sampling to the k most probable next tokens at each step. Lower values produce more focused output; higher values increase variety.                                 |
| top_p             | float (0–1)                               | Nucleus sampling: considers only tokens whose cumulative probability reaches p. Lower values make output more focused; higher values allow more diversity.                 |
| timeout_s         | integer                                   | How long to wait until the LLM request times out.                                                                                                                         |
| max_retries       | integer                                   | How many times to retry submitting an LLM message.                                                                                                                        |
| streaming         | true, false                               | When enabled, tokens stream to the chat UI in real time as they are generated. Disable for batch responses.                                                               |
| reasoning         | true, false or omitted                    | Enables extended reasoning/thinking mode on providers that support it (e.g., Anthropic extended thinking). Omit if unsupported.  