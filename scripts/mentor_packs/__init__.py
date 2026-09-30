"""Read-only evidence packs the Trade Mentor app hands its model. Pure and Qt-free.

A pack is one module exposing ``NAME``, ``SCHEMA`` (an Ollama tool schema),
``build(**args) -> Pack`` and ``fixture() -> Pack``. Every row carries an ``id`` the
model must cite. Packs read live stores and never write one.
"""
