if [ "$(uname)" == "Darwin" ]; then
  HOST_HOME='/Users'
else
  HOST_HOME='/home'
fi

docker run -it --rm \
  --runtime=nvidia --gpus all \
  -v ${HOST_HOME}/${USER}:/home/${USER} \
  --workdir /home/${USER} \
  -h ${HOSTNAME}-docker \
  --net=host $@ \
  ${USER}-precmae-cuda12.2
