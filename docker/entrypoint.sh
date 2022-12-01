#!/bin/bash

set -u

source ".venv/bin/activate"
COREF_SECRETS="/run/secrets/coref_secrets"
if [ -f "$COREF_SECRETS" ]; then
    export $(xargs < "$COREF_SECRETS");
fi

exec "$@"
