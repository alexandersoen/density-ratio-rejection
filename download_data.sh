#! /bin/bash

mkdir tmp
cd tmp

wget -N https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
unzip -o human+activity+recognition+using+smartphones.zip
unzip -o 'UCI HAR Dataset.zip'

mv 'UCI HAR Dataset' ../data/har

wget -N https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip
unzip -o gas+sensor+array+drift+dataset.zip
mv Dataset ../data/gas_drift

cd ..
rm -rf tmp
