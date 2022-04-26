# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

wget "https://zenodo.org/record/4282879/files/SemTab2020_Data.zip?download=1"
wget "https://zenodo.org/record/4246370/files/2T_WD.zip?download=1"

mkdir "${BASE_DATA_DIR}/SemTab/Benchmark/"
mv SemTab2020_Data.zip "${BASE_DATA_DIR}/SemTab/Benchmark/"
mv 2T_WD.zip "${BASE_DATA_DIR}/SemTab/Benchmark/"

cd "${BASE_DATA_DIR}/SemTab/Benchmark/"
unzip SemTab2020_Data.zip

cd SemTab2020_Data/SemTab2020_Table_GT_Target

unzip Round1/Tables_Round1.zip
unzip Round2/Tables_Round2.zip
unzip Round3/Tables_Round3.zip
unzip Round4/Tables_Round4.zip

cd "${BASE_DATA_DIR}/SemTab/Benchmark/"
unzip 2T_WD.zip