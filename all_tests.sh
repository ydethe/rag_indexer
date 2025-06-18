#! /bin/bash

mkdir -p htmldoc/ragindexer
rm -rf .root
.venv/bin/pytest
pdoc --html --force --config latex_math=True -o htmldoc ragindexer
coverage html -d htmldoc/coverage --rcfile tests/coverage.conf
coverage xml -o htmldoc/coverage/coverage.xml --rcfile tests/coverage.conf
docstr-coverage src/ragindexer -miP -sp -is -idel --skip-file-doc --badge=htmldoc/ragindexer/doc_badge.svg
genbadge coverage -l -i htmldoc/coverage/coverage.xml -o htmldoc/ragindexer/cov_badge.svg
