#!/bin/bash

#FINDOCDIR=../../findiff-docs/html
FINDOCDIR=./_build/html
DOCDIR=`pwd`

cd $FINDOCDIR || exit
git checkout gh-pages || exit

rm -rf $(find . -not -name . -not -name .. | grep -v .git)

cp -r ${DOCDIR}/_build/html/* .

git add .
git commit -m "Update documentation"

touch .nojekyll
git add .
git commit -m "Add .nojekyll file"

git push origin gh-pages

cd $DOCDIR