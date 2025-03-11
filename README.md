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

## Python Environment
On my system Python 3.10.12 and the following packages are installed:

Package                  Version <br>
absl-py                  2.1.0 <br>
accelerate               0.26.1 <br>
aiohttp                  3.9.1<br>
aiosignal                1.3.1<br>
alembic                  1.13.3<br>
async-timeout            4.0.3<br>
attrs                    21.2.0<br>
Automat                  20.2.0<br>
Babel                    2.8.0<br>
bcrypt                   3.2.0<br>
blinker                  1.4<br>
certifi                  2020.6.20<br>
chardet                  4.0.0<br>
charset-normalizer       3.3.2<br>
click                    8.0.3<br>
cloud-init               24.3.1<br>
colorama                 0.4.4<br>
colorlog                 6.8.2<br>
command-not-found        0.3<br>
configobj                5.0.6<br>
constantly               15.1.0<br>
contourpy                1.1.1<br>
cryptography             3.4.8<br>
cycler                   0.12.1<br>
datasets                 2.16.1<br>
dbus-python              1.2.18<br>
dill                     0.3.7<br>
distro                   1.7.0<br>
distro-info              1.1+ubuntu0.2<br>
evaluate                 0.4.1<br>
filelock                 3.12.4<br>
fonttools                4.43.1<br>
frozenlist               1.4.1<br>
fsspec                   2023.9.2<br>
greenlet                 3.1.1<br>
grpcio                   1.66.1<br>
httplib2                 0.20.2<br>
huggingface-hub          0.20.2<br>
hyperlink                21.0.0<br>
idna                     3.3<br>
importlib-metadata       4.6.4<br>
incremental              21.3.0<br>
jeepney                  0.7.1<br>
Jinja2                   3.0.3<br>
joblib                   1.3.2<br>
jsonpatch                1.32<br>
jsonpointer              2.0<br>
jsonschema               3.2.0<br>
kaleido                  0.2.1<br>
keyring                  23.5.0<br>
kiwisolver               1.4.5<br>
launchpadlib             1.10.16<br>
lazr.restfulclient       0.14.4<br>
lazr.uri                 1.0.6<br>
lxml                     5.1.0<br>
Mako                     1.3.5<br>
Markdown                 3.7<br>
MarkupSafe               2.1.5<br>
matplotlib               3.8.0<br>
more-itertools           8.10.0<br>
mpmath                   1.3.0<br>
multidict                6.0.4<br>
multiprocess             0.70.15<br>
netifaces                0.11.0<br>
networkx                 3.2<br>
nltk                     3.8.1<br>
numpy                    1.26.1<br>
nvidia-cublas-cu12       12.1.3.1<br>
nvidia-cuda-cupti-cu12   12.1.105<br>
nvidia-cuda-nvrtc-cu12   12.1.105<br>
nvidia-cuda-runtime-cu12 12.1.105<br>
nvidia-cudnn-cu12        8.9.2.26<br>
nvidia-cufft-cu12        11.0.2.54<br>
nvidia-curand-cu12       10.3.2.106<br>
nvidia-cusolver-cu12     11.4.5.107<br>
nvidia-cusparse-cu12     12.1.0.106<br>
nvidia-nccl-cu12         2.18.1<br>
nvidia-nvjitlink-cu12    12.2.140<br>
nvidia-nvtx-cu12         12.1.105<br>
oauthlib                 3.2.0<br>
optuna                   4.0.0<br>
packaging                23.2<br>
pandas                   2.1.4<br>
pexpect                  4.8.0<br>
Pillow                   10.1.0<br>
pip                      22.0.2<br>
plotly                   5.24.1<br>
portalocker              2.8.2<br>
protobuf                 5.28.0<br>
psutil                   5.9.7<br>
ptyprocess               0.7.0<br>
pyarrow                  14.0.2<br>
pyarrow-hotfix           0.6<br>
pyasn1                   0.4.8<br>
pyasn1-modules           0.2.1<br>
PyGObject                3.42.1<br>
PyHamcrest               2.0.2<br>
PyJWT                    2.3.0<br>
pyOpenSSL                21.0.0<br>
pyparsing                2.4.7<br>
pyrsistent               0.18.1<br>
pyserial                 3.5<br>
python-apt               2.4.0+ubuntu4<br>
python-dateutil          2.8.2<br>
python-debian            0.1.43+ubuntu1.1<br>
python-magic             0.4.24<br>
pytz                     2022.1<br>
PyYAML                   5.4.1<br>
regex                    2023.12.25<br>
requests                 2.31.0<br>
responses                0.18.0<br>
sacrebleu                2.4.0<br>
safetensors              0.4.1<br>
scikit-learn             1.5.2<br>
scipy                    1.14.1<br>
SecretStorage            3.3.1<br>
sentencepiece            0.1.99<br>
service-identity         18.1.0<br>
setuptools               59.6.0<br>
six                      1.16.0<br>
sos                      4.5.6<br>
SQLAlchemy               2.0.35<br>
ssh-import-id            5.11<br>
sympy                    1.12<br>
systemd-python           234<br>
tabulate                 0.9.0<br>
tenacity                 9.0.0<br>
tensorboard              2.17.1<br>
tensorboard-data-server  0.7.2<br>
threadpoolctl            3.5.0<br>
tokenizers               0.15.0<br>
torch                    2.1.0<br>
tqdm                     4.66.1<br>
transformers             4.36.2<br>
triton                   2.1.0<br>
Twisted                  22.1.0<br>
typing_extensions        4.8.0<br>
tzdata                   2023.4<br>
ubuntu-drivers-common    0.0.0<br>
ubuntu-pro-client        8001<br>
ufw                      0.36.1<br>
unattended-upgrades      0.1<br>
urllib3                  1.26.5<br>
wadllib                  1.3.6<br>
Werkzeug                 3.0.4<br>
wheel                    0.37.1<br>
xkit                     0.0.0<br>
xxhash                   3.4.1<br>
yarl                     1.9.4<br>
zipp                     1.0.0<br>
zope.interface           5.4.0<br>

## Citation
This code belongs to the following paper and should be cited as the same: 

Schmidt, M., Harbusch, K., & Memmesheimer, D. (2024, September). Automatic Ellipsis Reconstruction in Coordinated German Sentences Based on Text-to-Text Transfer Transformers. In International Conference on Text, Speech, and Dialogue (pp. 171-183). Cham: Springer Nature Switzerland.