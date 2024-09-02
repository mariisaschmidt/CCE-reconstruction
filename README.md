# Clausal Coordinate Ellipsis
This is the top level Readme. 
For more information have a look at src/README.md

For convenience we offer a bash script to simplify using our system. 

*NOTE:* this system requires CUDA, but it should be possible to deactivate this dependency, which will result in much longer runtimes. 

## Run the system
1. download the corpus files and place them in the src directory
2. make script executable: `chmod +x run.sh`
3. Execute `./run.sh --help`. This will help you set up the environment and train or evaluate models.

## Citation
This code belongs to the following paper and should be cited as the same: 

Schmidt, M., Harbusch, K., Memmesheimer, D. (2024). Automatic Ellipsis Reconstruction in Coordinated German Sentences Based on Text-to-Text Transfer Transformers. In: Nöth, E., Horák, A., Sojka, P. (eds) Text, Speech, and Dialogue. TSD 2024. Lecture Notes in Computer Science(), vol 15048. Springer, Cham. https://doi.org/10.1007/978-3-031-70563-2_14