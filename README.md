Implement of Paper [Monolingual Adapters for Zero-Shot Neural Machine Translation](https://aclanthology.org/2020.emnlp-main.361/)

## Getting Started

1. First you need to install fairseq according to the official fairseq documentation.

2. Prepare the dataset.

    - I used the **TED Talks** dataset in my reproduction of the paper. If you want this dataset too, please refer to the following link. [TED Talks](link.https://github.com/neulab/word-embeddings-for-nmt) You can use the ted_reader.py file to get the language pair data you need.
    
3. Train the multilingual transformer model.

    Firstly, you need to change the `finetune` argument to `False` in `fairseq/models/multilingual_transformer.py` file. Then you can train the model using the training script below.

    ```shell
    CUDA_VISIBLE_DEVICES=0 fairseq-train [data-bin path] \
        --max-epoch 100 \
        --ddp-backend=legacy_ddp \
        --task multilingual_translation --lang-pairs ar-en,he-en,it-en,de-en \
        --arch multilingual_transformer_iwslt_de_en \
        --share-decoders --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr 0.0005 --lr-scheduler inverse_sqrt \
        --warmup-updates 4000 --warmup-init-lr '1e-07' \
        --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
        --dropout 0.3 --weight-decay 0.0001 \
        --save-dir [checkpoint path] \
        --max-tokens 4000 \
        --update-freq 8 \
        --tensorboard-logdir /data2/lypan/fairseq/logs \
        --save-interval 5 --keep-best-checkpoints 1
    ```

4. Finetune the multilingual transformer using monolingual adapters

   Firstly, you need to change the `finetune` argument to `True` in `fairseq/models/multilingual_transformer.py` file. Then you can finetune the model using the finetuning script below.

   ```shell
   CUDA_VISIBLE_DEVICES=0 fairseq-train [data-bin path] \
       --max-epoch 120 \
       --ddp-backend=legacy_ddp \
       --task multilingual_translation --lang-pairs ar-en,he-en,it-en,de-en \
       --arch multilingual_transformer_iwslt_de_en \
       --share-decoders --share-decoder-input-output-embed \
       --optimizer adam --adam-betas '(0.9, 0.98)' \
       --lr 0.0005 --lr-scheduler inverse_sqrt \
       --warmup-updates 4000 --warmup-init-lr '1e-07' \
       --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
       --dropout 0.3 --weight-decay 0.0001 \
       --save-dir [checkpoint path] \
       --max-tokens 4000 \
       --update-freq 8 \
       --tensorboard-logdir /data2/lypan/fairseq/logs \
       --save-interval 5 --keep-best-checkpoints 1 \
       --restore-file [best checkpoint path(to load)]
   ```

## Result

Due to resource constraints, I only used four language pairs to train for this experiment(ar-en,he-en,it-en,de-en).

The results of the experiment are shown below:

|                          | ar-en | he-en | it-en | de-en | avg    |
|--------------------------|-------|-------|-------|-------|--------|
| multilingual transformer | 35.28 | 41.39 | 39.52 | 37.17 | 38.34  |
| +monolingual adaptation  | 35.26 | 41.86 | 39.83 | 37.77 | 38.68  |

As I am new to fairseq, there may still be some errors. If you find problems in using it, please contact me.