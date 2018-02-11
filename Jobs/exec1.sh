#PBS -l nodes=1
cd $PBS_O_WORKDIR
#this is for extraction of tamil dataset
echo "Extraction starting"
tar -xkvf Datasets/hpl-tamil-iso-char-offline-1.0.tar.gz -C Pickles/
echo "Extraction Done"
