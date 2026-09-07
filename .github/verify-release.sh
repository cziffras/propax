#!/usr/bin/env bash
# locally: .github/verify-release.sh [tag]
# check if tag name and toml version do match, if not reject
# the wheel suite is CI-only, it builds a venv and replays every test:
# force it locally with CI=1
set -euo pipefail

scratch=$(mktemp -d)
# precaution for local executions
trap 'rm -rf "$scratch"' EXIT

echo "-- remove egg and then build --"
# empty cache from previous versions, without it CI tests are not meaningful
rm -rf dist build src/*.egg-info
uv build

wheel=$(echo dist/*.whl)
tag="${1:-${GITHUB_REF_NAME:-}}"
declared=$(python3 -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

echo "-- version --"
if [ -z "$tag" ]; then
  echo "   no tag to check against, pyproject declares $declared"
elif [ "${tag#v}" = "$declared" ]; then
  echo "   tag ${tag#v} matches the declared version"
else
  echo "   tag ${tag#v} disagrees with the declared version $declared" >&2
  exit 1
fi

echo "-- metadata renders on PyPI --"
uv run --with twine twine check dist/*

echo "-- the sdist rebuilds and carries its data --"
# installing a sdist rebuilds it, which is the only path the wheel never takes
uv venv "$scratch/rebuilt"
uv pip install --python "$scratch/rebuilt/bin/python" dist/*.tar.gz
"$scratch/rebuilt/bin/python" -c "
from propax.fluids._registry import EQS_REGISTRY
assert EQS_REGISTRY, 'the rebuilt package ships no fluid'
print('   rebuilt, fluids:', sorted(EQS_REGISTRY))
"

if [ -z "${CI:-}" ]; then
  echo "-- skipping the wheel suite, CI is unset --"
  exit 0
fi

echo "-- the wheel passes the suite --"
uv venv "$scratch/wheelenv"
uv pip install --python "$scratch/wheelenv/bin/python" "${wheel}[test,coolprop]"
mkdir "$scratch/work" && cp -r tests pyproject.toml "$scratch/work/"
cd "$scratch/work" && "$scratch/wheelenv/bin/python" -m pytest tests -q
