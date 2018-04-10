#!/bin/bash

FINDOCDIR=../../findiff-docs/html
DOCDIR=`pwd`

cd $FINDOCDIR || exit
git checkout gh-pages || exit

cp -r ${DOCDIR}/_build/html/* .

git add .
git commit -m "Update documentation"
git push origin gh-pages

cd $DOCDIR