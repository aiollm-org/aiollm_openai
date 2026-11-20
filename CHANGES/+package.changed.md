**Breaking:** the OpenAI provider is its own package, `aiollm-openai`, imported
as `aiollm_openai`: `aiollm.providers.openai` becomes `aiollm_openai`, and each
of its modules keeps its name (`aiollm.providers.openai.types` becomes
`aiollm_openai.types`, and so on). `aiollm_openai.__version__` is its own
version.
