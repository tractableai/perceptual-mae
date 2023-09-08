docker build \
  --build-arg USERNAME=$USER \
  --build-arg USER_UID=$UID \
  -t ${USER}-precmae-cuda12.2 .