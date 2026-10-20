# Griptape Nodes Research Labs Libraries

Griptape Nodes libraries from the Research Labs team. Each library lives in its own folder, with its own README, dependencies and version.

## Libraries

| Library | Folder | Description |
|---------|--------|-------------|
| Face Parsing | [`face_parsing/`](face_parsing/README.md) | Segments a face into 19 parts with BiSeNet, producing a mask and a color overlay. |

## Adding a library

Each library folder is self-contained, with its own `pyproject.toml`, `Makefile`, `uv.lock`, and a manifest under its package directory. Add a new library by creating a sibling folder and listing it here.

## License

Apache License 2.0. See [LICENSE](LICENSE). Each library's README notes any bundled third-party code and its license.
