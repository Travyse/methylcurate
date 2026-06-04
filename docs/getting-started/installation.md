# Installation

## (1) Prerequisites
 * **Docker**: Install Docker and ensure the Docker daemon is running.
 * **LLM Configuration File**: Provide a `.yml` file with LLM credentials and parameters. See the LLM Configuration section below for more details.

## (2) Configure LLM

Copy and edit the example LLM configuration file:

```bash
cp llm_config.example.yml llm_config.yml
# Edit llm_config.yml with your provider, model, and credentials
```

API keys can be set directly in the config file or via environment variables. The example config uses `${VAR}` syntax so you can keep secrets out of the file:

```bash
export OPENAI_API_KEY=sk-...
```

## (3) Build Docker Image

Download the latest MethylCurate repo and build the Docker image.

```bash
git clone git@github.com:travyse/methylcurate.git
cd methylcurate
docker compose build
docker compose up
```

The `docker compose build` command builds the Docker image, while `docker compose up` runs the image. Each time you want to use the tool, you only need to run `docker compose up`.

All workflow outputs (metadata, beta matrices, clock results, and logs) are saved to the `outputs` Docker volume and can be accessed at `./outputs` on the host.

## (4) Launching Tool

To launch the tool, open your browser of interest and visit [http://localhost:3000](http://localhost:3000) to use the tool.