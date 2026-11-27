# Scaleway Generative APIs

**Author:** scaleway
**Version:** 0.0.1
**Type:** model
**Source:** https://github.com/scaleway/scaleway-dify-plugin

Use models from [Scaleway Generative APIs](https://www.scaleway.com/en/docs/generative-apis/quickstart/) in Dify:

- **Chat** models, with tool calling, vision and structured output where the model supports them
- **Text embedding** models, for Knowledge bases
- **Speech-to-text** (Whisper), for voice input

## Setup

1. Create a Scaleway API key with access to Generative APIs: see [Create API keys](https://www.scaleway.com/en/docs/iam/how-to/create-api-keys/).
2. In Dify, go to **Integrations** > **Model Provider**, find **Scaleway**, click **Add API Key**, and enter the key's **secret key**.
3. Optional: to bill a specific Scaleway project, enter its **Project ID**. Leave it empty to use your default project.

## Models

The plugin ships Scaleway's current Generative APIs models. Scaleway adds and retires models over time; to use one that isn't listed, click **Add Model** on the Scaleway card and enter its ID. List the current IDs with:

```bash
curl https://api.scaleway.ai/v1/models -H "Authorization: Bearer $SCW_SECRET_KEY"
```

See [Supported models](https://www.scaleway.com/en/docs/generative-apis/reference-content/supported-models/) for each model's context size, maximum output and capabilities.

Models added with **Add Model** have their own credentials, so enter the API key (and Project ID, if any) again in that form.

### Dedicated deployments

**Add Model** also works with [dedicated deployments](https://www.scaleway.com/en/docs/managed-inference/):

- **API Base URL:** the deployment's endpoint from the Scaleway console, with `/v1` added at the end, for example `https://<DEPLOYMENT_ID>.ifr.fr-par.scaleway.com/v1`.
- **Model Name:** the model name the deployment reports at its `/v1/models` endpoint.
- **Project ID:** leave empty; the API Base URL already identifies the deployment.

## Usage

- **Chat:** select a Scaleway model in any app or LLM node.
- **Knowledge:** in **Model Provider**, click **Default Models** and set a Scaleway embedding model as the **Embedding Model**.
- **Voice input:** in **Default Models**, set **Whisper Large v3** as the **Speech-to-Text Model**. Then turn on **Speech to Text** in your app's **Features**, and publish the app.