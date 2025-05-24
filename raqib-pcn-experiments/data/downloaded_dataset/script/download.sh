#!/bin/bash

base_url="https://zenodo.org/record/6619768/files"
destination="../jetclass_dataset"
mkdir -p "$destination"

files=(
  "JetClass_Pythia_train_100M_part0.tar"
  "JetClass_Pythia_train_100M_part1.tar"
  "JetClass_Pythia_train_100M_part2.tar"
  "JetClass_Pythia_train_100M_part3.tar"
  "JetClass_Pythia_train_100M_part4.tar"
  "JetClass_Pythia_train_100M_part5.tar"
  "JetClass_Pythia_train_100M_part6.tar"
  "JetClass_Pythia_train_100M_part7.tar"
  "JetClass_Pythia_train_100M_part8.tar"
  "JetClass_Pythia_train_100M_part9.tar"
  "JetClass_Pythia_val_5M.tar"
  "JetClass_Pythia_test_20M.tar"
)

# MD5 checksums from Zenodo
declare -A checksums
checksums["JetClass_Pythia_train_100M_part0.tar"]="de4fd2dca2e68ab3c85d5cfd3bcc65c3 "
checksums["JetClass_Pythia_train_100M_part1.tar"]="9722a359c5ef697bea0fbf79bf50f003  "
checksums["JetClass_Pythia_train_100M_part2.tar"]="1e9f66cd1f915f9d10e90ae1d7761720 "
checksums["JetClass_Pythia_train_100M_part3.tar"]="47348fc8985319fa4806da87500482fa"
checksums["JetClass_Pythia_train_100M_part4.tar"]="6b0ce16bd93b442a8d51914466990279 "
checksums["JetClass_Pythia_train_100M_part5.tar"]="416e347512e716de51d392bee327b8e9 "
checksums["JetClass_Pythia_train_100M_part6.tar"]="e9b9c1557b1b39bf0a16e4ab631ae451 "
checksums["JetClass_Pythia_train_100M_part7.tar"]="5bfc6cb285ccb7680cefa9ac82ad1a2e "
checksums["JetClass_Pythia_train_100M_part8.tar"]="540c1a0d66dfad78d2b363c5740ccf86 "
checksums["JetClass_Pythia_train_100M_part9.tar"]="668f40b3275167ff7104c48317c0ae2a "
checksums["JetClass_Pythia_val_5M.tar"]="7235ccb577ed85023ea3ab4d5e6160cf"
checksums["JetClass_Pythia_test_20M.tar"]="64e5156d26d101adeb43b8388207d767"



for file in "${files[@]}"; do
    echo "Downloading $file to $destination..."
    wget -c "${base_url}/${file}?download=1" -O "${destination}/${file}"

    echo "Verifying checksum for $file..."
    local_md5=$(md5sum "${destination}/${file}" | awk '{print $1}')
    expected_md5="${checksums[$file]}"

    if [[ "$local_md5" == "$expected_md5" ]]; then
        echo "✅ Checksum matched for $file"
    else
        echo " Checksum mismatch for $file! File may be corrupted."
    fi
done
