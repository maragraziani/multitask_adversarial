Help()
{
   # Display Help
   echo "Uncertainty Weighted Multi-task Adversarial Architecture"
   echo
   echo "Usage: "
   echo "train_uncertainty_weighted_mta.sh [EXPERIMENT_TYPE] [SEED] [CONCEPT_LIST]"
   echo
   echo "Available concepts: "
   echo "domain: WSI acquisition center"
   echo "narea: nuclei area"
   echo "ncount: nuclei count"
   echo "nuclei_correlation: texture correlation inside the nuclei"
   echo "nuclei_contrast: texture contrast inside the nuclei"
   echo "full_correlation: full image texture correlation"
   echo "full_contrast: full image texture contrast"
   echo
}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;
   esac
done
#upgraded to handle multiple concepts at a time ?!
#concepts=['full_contrast','full_correlation','narea', 'ncount','nuclei_contrast','nuclei_correlation']
EXPERIMENT_TYPE=$1
SEED=$2
CONCEPT_LIST=[$3]
FILE=results/$EXPERIMENT_TYPE
if [[ -d $FILE ]]
then
    read -r -p "override results in folder? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
    then
        #horovodrun -np 2 \
        #-H localhost:4 \
        python scripts/train_multitask_adversarial.py $SEED $EXPERIMENT_TYPE $CONCEPT_LIST
    else
        echo Stopping process to avoid overriding data.
    fi
else
    mkdir results/$EXPERIMENT_TYPE
    #horovodrun -np 2 \
    #    -H localhost:4 \
        python scripts/train_multitask_adversarial.py $SEED $EXPERIMENT_TYPE $CONCEPT_LIST
fi

