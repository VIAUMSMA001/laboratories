# AI Technologies (VIAUMSMA001) Laboratories

Laboratory materials and assignments for the [VIAUMSMA001 - AI Technologies](https://portal.vik.bme.hu/kepzes/targyak/VIAUMSMA001-00/) course.

The notes are built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and published via GitHub Pages.

## Running the docs locally (with Docker)

This repository contains a `Dockerfile` with the MkDocs framework and its dependencies. Build the image once, then run it to serve the documentation with hot reload.

1. Open a terminal in the root of this project.
2. Build and run the container:

    ```bash
    docker build -t mkdocs .
    docker run -it --rm -p 8000:8000 -v ${PWD}:/docs mkdocs
    ```

3. Open <http://localhost:8000> in your browser.
4. Editing and saving a markdown file triggers a hot reload.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose fixes or additions to the material.
