# Data Preprocessing

1. Download the FEVER data and Wikipedia dump ('Pre-processed Wikipedia Pages, June 2017 dump') from https://fever.ai/resources.html. Run the code indicated in [scripts/data/fever_data_preprocessing.sh](scripts/data/fever_data_preprocessing.sh) after updating directories, as applicable.

2. Download the Symmetric analysis sets from https://github.com/TalSchuster/FeverSymmetric and run [scripts/data/symmetric_data_preprocessing.sh](scripts/data/symmetric_data_preprocessing.sh) for the desired subsets. Note that the original FEVER data is also needed in order to re-associate the Wikipedia title and sentence number to each of the evidence sentences.
