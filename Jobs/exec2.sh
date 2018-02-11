#PBS -l nodes=1
cd  ~/TamilCharacterRecognition/Pickles
pwd
python3 CountData.py tamil_dataset_offline
echo "DOne"
