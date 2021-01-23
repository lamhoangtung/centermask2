CUDA_HOME=/ python3 train_net.py --config-file "configs/centermask/publaynet_centermask_lite_V_19_eSE_FPN_ms_4x.yaml" --num-gpus 1 \
	MODEL.ROI_HEADS.NUM_CLASSES 5 \
       	MODEL.SEM_SEG_HEAD.NUM_CLASSES 5 \
        MODEL.RETINANET.NUM_CLASSES 5 \
        MODEL.FCOS.NUM_CLASSES 5 \
	TEST.EVAL_PERIOD 1000
