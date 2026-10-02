#!/usr/bin/env bash
# Clone the two open-source repos this example's tasks were mined from.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p repos
[ -d repos/itsdangerous ] || git clone https://github.com/pallets/itsdangerous.git repos/itsdangerous
[ -d repos/python-slugify ] || git clone https://github.com/un33k/python-slugify.git repos/python-slugify
echo "Done. Test dependencies: pip install pytest freezegun text-unidecode"
