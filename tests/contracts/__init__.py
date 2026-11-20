"""Snapshots of OpenAI's wire contracts and the script that refreshes them.

Each ``<endpoint>/`` folder holds the upstream JSON schemas of one
endpoint (request, response and, with a stream, its events) and the
examples that pass them. ``manifest.json`` records where every schema
came from. ``test_wire_contracts.py`` checks the library against them;
``update.py`` rebuilds them from OpenAI's official OpenAPI file.
"""
