PYG_WORKSPACE=$PWD
USER=sykang
PY_EXEC="python3"
EXEC_SCRIPT="${PYG_WORKSPACE}/test.py"
CMD="cd ${PYG_WORKSPACE}; ${PY_EXEC} ${EXEC_SCRIPT}"

NUM_NODES=2

DATASET=ogbn-products

DATASET_ROOT_DIR="./data/partitions/${DATASET}/${NUM_NODES}=parts"

IP_CONFIG=${PYG_WORKSPACE}/ip_config.yaml

python3 launch.py --workspace ${PYG_WORKSPACE} --ip_config ${IP_CONFIG} --ssh_username ${USER} --num_nodes ${NUM_NODES} --dataset_root_dir ${DATASET_ROOT_DIR} --dataset ${DATASET}

echo "started launch.py: ${pid}"

trap "kill -2 $pid" SIGINT
wait $pid
set +x