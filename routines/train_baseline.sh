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

