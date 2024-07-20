# Defending Graph Neural Networks against Adversarial Attacks 
This repository was used in the  <a href="">Molecular Property Prediction Project</a>. The pre-trained backbone can be extended for several other uses cases as well. 

## Running the Scripts
- Follow the procedure detailed <a href="https://github.com/Deceptrax123/Molecular-Graph-Featuriser" >here</a> to save the tensors of all molecular graphs.
- You may use the SMILE strings from either Zinc or TwoSides. Note that TwoSides has 2 molecules per instance and different pairing and batching strategies need to be used to successfully run the scripts. The procedure for such a usecase is detailed <a href="https://github.com/Deceptrax123/Drug-Drug-Interaction-Dataloader">here</a>
- Run the following command:
  ```sh
  pip install -r requirements.txt
  ```
- Set the values for the paths in a ```.env``` file. The keys are detailed below.
- Run ```train.py``` for the dataset of your choice. Further, you may be required to re-set the environment path. For such cases, linux/macOS users may use:
  ```sh
  export PYTHONPATH="/path/to/project/root"
  ```
- Windows users are suggested to use ```bash``` terminal and run the above command.


## Datasets Used
- ZINC
- TwoSIDES

## Environment Variables
|Key|Value|
|---|------|
|graph_files|path/to/graphs/data/processed/|

You may send an email or raise an issue if there are any bugs.