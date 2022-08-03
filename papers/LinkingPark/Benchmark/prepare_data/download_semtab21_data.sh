benchmark_dir="${BASE_DATA_DIR_SEMTAB21}/SemTab/Benchmark/"
mkdir ${benchmark_dir}
cd ${benchmark_dir}

echo "Processing Round1 ..."

mkdir Round1
wget https://www.cs.ox.ac.uk/isg/challenges/sem-tab/2021/data/tables_CTA_CEA_WD_Round1.tar.gz
tar -xvzf tables_CTA_CEA_WD_Round1.tar.gz

wget https://www.cs.ox.ac.uk/isg/challenges/sem-tab/2021/data/CEA_WD_Round1_Targets.csv
wget https://www.cs.ox.ac.uk/isg/challenges/sem-tab/2021/data/CTA_WD_Round1_Targets.csv

mkdir Round1/tables
mkdir Round1/targets

mv tables_WD/* Round1/tables
mv "CEA_WD_Round1_Targets.csv" Round1/targets
mv "CTA_WD_Round1_Targets.csv" Round1/targets
rm -rf tables_WD/
rm tables_CTA_CEA_WD_Round1.tar.gz

echo "Processing HardTables ..."

wget -O SemTab2021_HardTables.zip https://zenodo.org/record/6154708/files/SemTab2021_HardTables.zip?download=1
unzip SemTab2021_HardTables.zip

echo "Processing Round2 HardTables ..."
mkdir Round2/HardTable
mkdir Round2/HardTable/gt Round2/HardTable/tables Round2/HardTable/targets
mv Round2/HardTablesR2/* Round2/HardTable/tables
mv Round2/HardTablesR2_CEA_WD/HardTablesR2_CEA_WD_gt.csv Round2/HardTable/gt
mv Round2/HardTablesR2_CTA_WD/HardTablesR2_CTA_WD_gt.csv Round2/HardTable/gt
mv Round2/HardTablesR2_CTA_WD/HardTablesR2_CTA_WD_gt_ancestor.json Round2/HardTable/gt
mv Round2/HardTablesR2_CTA_WD/HardTablesR2_CTA_WD_gt_descendent.json Round2/HardTable/gt
mv Round2/HardTablesR2_CPA_WD/HardTablesR2_CPA_WD_gt.csv Round2/HardTable/gt
mv Round2/HardTablesR2_CEA_WD/HardTableR2_CEA_WD_Round2_Targets.csv Round2/HardTable/targets/HardTablesR2_CEA_WD_Round2_Targets.csv
mv Round2/HardTablesR2_CTA_WD/HardTablesR2_CTA_WD_Round2_Targets.csv Round2/HardTable/targets/HardTablesR2_CTA_WD_Round2_Targets.csv
mv Round2/HardTablesR2_CPA_WD/HardTableR2_CPA_WD_Round2_Targets.csv Round2/HardTable/targets/HardTablesR2_CPA_WD_Round2_Targets.csv
rm -rf Round2/HardTablesR2/
rm -rf Round2/HardTablesR2_CEA_WD/
rm -rf Round2/HardTablesR2_CTA_WD/
rm -rf Round2/HardTablesR2_CPA_WD/

echo "Processing Round3 HardTables ..."
mkdir Round3/HardTable
mkdir Round3/HardTable/gt Round3/HardTable/tables Round3/HardTable/targets
mv Round3/HardTablesR3/* Round3/HardTable/tables
mv Round3/HardTablesR3_CEA_WD/HardTablesR3_CEA_WD_gt.csv Round3/HardTable/gt
mv Round3/HardTablesR3_CTA_WD/HardTablesR3_CTA_WD_gt.csv Round3/HardTable/gt
mv Round3/HardTablesR3_CTA_WD/HardTablesR3_CTA_WD_gt_ancestor.json Round3/HardTable/gt
mv Round3/HardTablesR3_CTA_WD/HardTablesR3_CTA_WD_gt_descendent.json Round3/HardTable/gt
mv Round3/HardTablesR3_CPA_WD/HardTablesR3_CPA_WD_gt.csv Round3/HardTable/gt
mv Round3/HardTablesR3_CEA_WD/HardTablesR3_CEA_WD_Round3_Targets.csv Round3/HardTable/targets
mv Round3/HardTablesR3_CTA_WD/HardTablesR3_CTA_WD_Round3_Targets.csv Round3/HardTable/targets
mv Round3/HardTablesR3_CPA_WD/HardTablesR3_CPA_WD_Round3_Targets.csv Round3/HardTable/targets
rm -rf Round3/HardTablesR3/
rm -rf Round3/HardTablesR3_CEA_WD/
rm -rf Round3/HardTablesR3_CTA_WD/
rm -rf Round3/HardTablesR3_CPA_WD/

rm SemTab2021_HardTables.zip

echo "Processing Round2 BioTable ..."

wget -O BioTable-Datasets.zip https://zenodo.org/record/5606585/files/BioTable-Datasets.zip?download=1
unzip BioTable-Datasets.zip
mkdir -p Round2/BioTable/tables
mkdir -p Round2/BioTable/targets
mkdir -p Round2/BioTable/gt
mv datasets/* Round2/BioTable/tables
mv ground-truth/cea_gt.csv Round2/BioTable/gt
mv ground-truth/cta_gt.csv Round2/BioTable/gt
mv ground-truth/cpa_gt.csv Round2/BioTable/gt
mv ground-truth/cea_gt_targets.csv Round2/BioTable/targets/BioTable_CEA_WD_Round2_Targets.csv
mv ground-truth/cta_gt_targets.csv Round2/BioTable/targets/BioTable_CTA_WD_Round2_Targets.csv
mv ground-truth/cpa_gt_targets.csv Round2/BioTable/targets/BioTable_CPA_WD_Round2_Targets.csv
rm BioTable-Datasets.zip
rm -rf datasets/
rm -rf ground-truth/

echo "Processing Round3 BioDivTable ..."
wget -O BioDivTab.zip https://zenodo.org/record/5584180/files/fusion-jena/BiodivTab-v0.1_2021.zip?download=1
unzip BioDivTab.zip
mkdir -p Round3/BioDivTab
mv fusion-jena-BiodivTab-5191adf/* Round3/BioDivTab
rm BioDivTab.zip
rm -rf fusion-jena-BiodivTab-5191adf/

echo "Done."