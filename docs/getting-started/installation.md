# Installation

## (1) Prerequisites
 * **Docker**: Install Docker and ensure the Docker daemon is running.
 * **LLM Configuration File**: Provide a `.yml` file with LLM credentials and parameters. See the LLM Configuration section below for more details.

## (2) Configure LLM

Copy and edit the example LLM configuration file. In the following example, we name it `llm_config.yml`:

```bash
cp llm_config.example.yml llm_config.yml
```

API keys can be set directly in the config file or via environment variables. The example config uses `${VAR}` syntax so you can keep secrets out of the file:

```bash
export OPENAI_API_KEY=sk-...
```

## (3) Configure Quality Control

This is optional, but you can copy and edit the quality control configuration file. In the following example, we name it `qc_config.yml`:

```bash
cp qc_config.example.yml qc_config.yml\
```

## (4) Modify Compose File

Modify the `compose.yaml` file to specify output directories and set the environment variables to point to your LLM and quality control config files.

## (5) Build Docker Image

Download the latest MethylCurate repo and build the Docker image.

```bash
git clone git@github.com:travyse/methylcurate.git
cd methylcurate
docker compose -f compose.yaml build
docker compose -f compose.yaml up
```

The `docker compose -f compose.yaml build` command builds the Docker image, while `docker compose -f compose.yaml up` runs the image. The `-f` flag is used to specify the specific compose file to use. Each time you want to use the tool, you only need to run `docker compose -f compose.yaml up`.

All workflow outputs (metadata, beta matrices, clock results, and logs) are saved to the `outputs` Docker volume and can be accessed at `./outputs` on the host by default.

## (6) Launching Tool

To launch the tool, open your browser of interest and visit [http://localhost:3000](http://localhost:3000) to use the tool.

### Browsers Tested

We have successfully tested MethylCurate with Safari version 18.1.1 and Google Chrome version 148.0.7778.216. Other browsers may not support our tool.