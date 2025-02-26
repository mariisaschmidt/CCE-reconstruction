# Clausal Coordinate Ellipsis Reconstruction in German
This is the top level Readme. 
For more information have a look at src/README.md

## Project Structure
- CCE-reconstruction/
  - data/                   -> Directory for corpus files
  - models/                 -> Directory for model files
  - scripts/                -> Directory for bash scripts
  - src/                    -> Directory for source code
  - visualizations/         -> Directory for visualization notebooks
  - run.sh                  -> Bash script to set up and run the system
  - README.md               -> Top level Readme

## Quickly run the system
For convenience we offer a bash script to simplify using our system. 

1. download the corpus files and place them in the data directory
2. make script executable: `chmod +x run.sh`
3. Execute `./run.sh --help`. This will help you set up the environment and train or evaluate models.

*NOTE:* this system requires CUDA, but it should be possible to deactivate this dependency, which will result in much longer runtimes. 

## Citation
This code belongs to the following paper and should be cited as the same: 

Schmidt, M., Harbusch, K., & Memmesheimer, D. (2024, September). Automatic Ellipsis Reconstruction in Coordinated German Sentences Based on Text-to-Text Transfer Transformers. In International Conference on Text, Speech, and Dialogue (pp. 171-183). Cham: Springer Nature Switzerland.