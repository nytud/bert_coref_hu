FROM tensorflow/tensorflow:2.9.1-gpu AS build

WORKDIR /app
COPY ./pyproject.toml ./poetry.lock ./
COPY ./src/ ./src/

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.2.0
ENV PATH="${PATH}:/root/.local/bin"

RUN poetry config virtualenvs.in-project true && \
    poetry config virtualenvs.options.system-site-packages true && \
    poetry install

FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR /app
COPY . .
COPY --from=build /app/.venv/ /app/.venv/
ENTRYPOINT ["./docker/entrypoint.sh"]
RUN chmod 775 ./docker/entrypoint.sh

ARG DATA_DIR="/data"
ARG UID=""
ARG GID=""
RUN if ! [ -d ${DATA_DIR} ]; then mkdir ${DATA_DIR}; fi
RUN if ! [ -z ${UID} ] && ! [ -z ${GID} ]; \
    then mkdir /.cache /.cache/huggingface; \
    chown -R ${UID}:${GID} /.cache/huggingface; \
    chown -R ${UID}:${GID} ${DATA_DIR}; fi
