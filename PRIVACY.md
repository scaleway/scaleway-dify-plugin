# Privacy Policy

This plugin connects Dify to Scaleway Generative APIs.

## Data sent to Scaleway

When you use a Scaleway model in Dify, the plugin sends the following to Scaleway's API (`api.scaleway.ai`, or the base URL you configure):

- **Chat:** the messages, any images, and tool definitions in the request
- **Text embedding:** the texts to embed, such as chunks of documents added to a Knowledge base
- **Speech-to-text:** the audio file to transcribe

Requests are authenticated with the Scaleway API key you provide. Scaleway processes this data under its own terms and privacy policy: https://www.scaleway.com/en/privacy-policy/

## Data stored by the plugin

The plugin itself does not store, log or share any of this data. Your API key is stored by Dify as an encrypted credential.

## Contact

For questions about this plugin, open an issue at https://github.com/scaleway/scaleway-dify-plugin/issues.