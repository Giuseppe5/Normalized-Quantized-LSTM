#!bin/bash

docker run -it --rm \
--runtime=nvidia \
--shm-size=8g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /scratch/users/gfranco/brevitas_fork:/brevitas \
-v /scratch/users/gfranco/Normalized-Quantized-LSTM:/Normalized-Quantized-LSTM \
gfranco/lstm_quant bash
