Use updated deepsignal_custom script to run the model
Main change: `deepsignal_custom/deepsignal/model.py`
1. Use softmax for multiple-class instead of sigmoid for binary classification
2. Change loss funcation: tf.nn.softmax_cross_entropy_with_logits

(base) [c-panz@winter-log2 2021-11-29]$ squeue -u c-panz
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            220558       gpu 512000_t   c-panz  R       0:01      1 winter203
            220557       gpu train_mo   c-panz  R       7:21      1 winter201