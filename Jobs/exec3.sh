#PBS -l nodes=1
cd $PBS_O_WORKDIR

cd ../
echo " Training Model"
python trainModel.py 100
echo "Model Trained"

