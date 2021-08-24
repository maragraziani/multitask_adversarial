Help()
{
   # Display Help
   echo "Baseline Architecture (without extra branches)"
   echo
   echo "Usage: "
   echo "train_baseline.sh [SEED] [EXPERIMENT_TYPE]"
   echo
}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;
   esac
done

EXPERIMENT_TYPE=$1
SEED=$2
FILE=results/$EXPERIMENT_TYPE
if [[ -d $FILE ]]
then
    read -r -p "override results in folder? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
    then
        #horovodrun -np 2 \
        #-H localhost:4 \
        python scripts/train_baseline.py $SEED $EXPERIMENT_TYPE
    else
        echo Stopping process to avoid overriding data.
    fi
else
    mkdir results/$EXPERIMENT_TYPE
    #horovodrun -np 2 \
        #-H localhost:4 \
        python scripts/train_baseline.py $SEED $EXPERIMENT_TYPE
fi

